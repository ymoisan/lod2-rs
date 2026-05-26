use lod2_common::polygon::{AttributeMap, Footprint, LinearRing, Polygon3D};
use nalgebra::Point3;
use rusqlite::Connection;
use std::path::Path;

/// Read building footprints from a GeoPackage file using pure-Rust SQLite.
pub fn read_footprints(path: &Path) -> anyhow::Result<Vec<Footprint>> {
    anyhow::ensure!(path.exists(), "File not found: {}", path.display());

    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // Find the first feature table and its geometry column.
    let (table, geom_col): (String, String) = conn.query_row(
        "SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    // Discover non-geometry column names and types for attribute reading.
    let mut attr_cols: Vec<String> = Vec::new();
    {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info(\"{}\")", table))?;
        let rows = stmt.query_map([], |row| {
            let name: String = row.get(1)?;
            Ok(name)
        })?;
        for name in rows {
            let name = name?;
            if name != geom_col && name != "fid" {
                attr_cols.push(name);
            }
        }
    }

    // Build SELECT: fid, geometry, attr1, attr2, ...
    let attr_select = attr_cols
        .iter()
        .map(|c| format!("\"{}\"", c))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = if attr_cols.is_empty() {
        format!("SELECT fid, \"{}\" FROM \"{}\"", geom_col, table)
    } else {
        format!(
            "SELECT fid, \"{}\", {} FROM \"{}\"",
            geom_col, attr_select, table
        )
    };

    let mut stmt = conn.prepare(&sql)?;
    let mut footprints = Vec::new();
    let mut rows = stmt.query([])?;

    while let Some(row) = rows.next()? {
        let fid: i64 = row.get(0)?;
        let geom_blob: Vec<u8> = row.get(1)?;

        let polygon = match parse_gpkg_geometry(&geom_blob) {
            Some(p) if !p.is_empty() => p,
            _ => continue,
        };

        let mut attributes = AttributeMap::new();
        for (i, col_name) in attr_cols.iter().enumerate() {
            let col_idx = i + 2; // offset past fid and geom
            // Try each type; SQLite is dynamically typed.
            if let Ok(v) = row.get::<_, i64>(col_idx) {
                attributes.insert_int(col_name, v);
            } else if let Ok(v) = row.get::<_, f64>(col_idx) {
                attributes.insert_float(col_name, v);
            } else if let Ok(v) = row.get::<_, String>(col_idx) {
                attributes.insert_string(col_name, v);
            }
            // NULL or unsupported types silently skipped
        }

        footprints.push(Footprint::new(fid.to_string(), polygon).with_attributes(attributes));
    }

    tracing::info!("Read {} footprints from {}", footprints.len(), path.display());
    Ok(footprints)
}

/// Read CRS (EPSG code) from a GeoPackage file.
pub fn read_crs(path: &Path) -> anyhow::Result<Option<String>> {
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // Get the SRS ID from the geometry columns table.
    let srs_id: i64 = conn.query_row(
        "SELECT srs_id FROM gpkg_geometry_columns LIMIT 1",
        [],
        |row| row.get(0),
    )?;

    // Look up the EPSG code in the spatial reference system table.
    let result: rusqlite::Result<(String, i64)> = conn.query_row(
        "SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id = ?1",
        [srs_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    match result {
        Ok((org, code)) if org.eq_ignore_ascii_case("EPSG") => {
            tracing::info!("Found CRS: EPSG:{}", code);
            Ok(Some(code.to_string()))
        }
        _ => Ok(None),
    }
}

// ── GeoPackage Binary + WKB parsing ──

/// Parse a GeoPackage Binary geometry blob into a Polygon3D.
///
/// GeoPackage Binary format:
///   - 2 bytes: magic "GP"
///   - 1 byte: version
///   - 1 byte: flags (bits 1-3 = envelope type, bit 0 = byte order)
///   - 4 bytes: SRS ID
///   - envelope (size depends on flags)
///   - standard WKB payload
fn parse_gpkg_geometry(blob: &[u8]) -> Option<Polygon3D> {
    if blob.len() < 8 || blob[0] != b'G' || blob[1] != b'P' {
        return None;
    }

    let flags = blob[3];
    let byte_order = flags & 0x01; // 0 = big-endian, 1 = little-endian
    let envelope_type = (flags >> 1) & 0x07;

    // Envelope sizes: 0=none, 1=xy(32), 2=xyz(48), 3=xym(48), 4=xyzm(64)
    let envelope_size = match envelope_type {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return None,
    };

    let wkb_offset = 8 + envelope_size;
    if blob.len() <= wkb_offset {
        return None;
    }

    parse_wkb_polygon(&blob[wkb_offset..], byte_order)
}

/// Parse a WKB geometry, handling Polygon and MultiPolygon types.
fn parse_wkb_polygon(wkb: &[u8], _parent_byte_order: u8) -> Option<Polygon3D> {
    if wkb.len() < 5 {
        return None;
    }

    let byte_order = wkb[0]; // Each WKB geometry has its own byte order
    let geom_type = read_u32(&wkb[1..5], byte_order);

    match geom_type {
        // Polygon, PolygonZ, PolygonM, PolygonZM
        3 | 1003 | 2003 | 3003 => parse_wkb_polygon_rings(&wkb[5..], byte_order, has_z(geom_type)),
        // MultiPolygon, MultiPolygonZ — take first polygon
        6 | 1006 => {
            if wkb.len() < 9 {
                return None;
            }
            let _num_geoms = read_u32(&wkb[5..9], byte_order);
            // Recurse into the first polygon geometry
            parse_wkb_polygon(&wkb[9..], byte_order)
        }
        _ => None,
    }
}

fn has_z(geom_type: u32) -> bool {
    matches!(geom_type, 1003 | 3003 | 1006)
}

fn parse_wkb_polygon_rings(data: &[u8], byte_order: u8, z: bool) -> Option<Polygon3D> {
    let mut offset = 0;
    if data.len() < 4 {
        return None;
    }
    let num_rings = read_u32(&data[offset..offset + 4], byte_order) as usize;
    offset += 4;

    if num_rings == 0 {
        return Some(Polygon3D::default());
    }

    let coord_size = if z { 24 } else { 16 }; // 2 or 3 f64s

    let mut rings = Vec::with_capacity(num_rings);
    for _ in 0..num_rings {
        if offset + 4 > data.len() {
            return None;
        }
        let num_points = read_u32(&data[offset..offset + 4], byte_order) as usize;
        offset += 4;

        let needed = num_points * coord_size;
        if offset + needed > data.len() {
            return None;
        }

        let mut vertices = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let x = read_f64(&data[offset..offset + 8], byte_order);
            let y = read_f64(&data[offset + 8..offset + 16], byte_order);
            let z_val = if z {
                read_f64(&data[offset + 16..offset + 24], byte_order)
            } else {
                0.0
            };
            vertices.push(Point3::new(x, y, z_val));
            offset += coord_size;
        }
        rings.push(LinearRing::from_vertices(vertices));
    }

    let exterior = rings.remove(0);
    Some(Polygon3D::with_interiors(exterior, rings))
}

fn read_u32(data: &[u8], byte_order: u8) -> u32 {
    let bytes: [u8; 4] = [data[0], data[1], data[2], data[3]];
    if byte_order == 1 {
        u32::from_le_bytes(bytes)
    } else {
        u32::from_be_bytes(bytes)
    }
}

fn read_f64(data: &[u8], byte_order: u8) -> f64 {
    let bytes: [u8; 8] = [
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ];
    if byte_order == 1 {
        f64::from_le_bytes(bytes)
    } else {
        f64::from_be_bytes(bytes)
    }
}
