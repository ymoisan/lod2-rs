use crate::polygon::{AttributeValue, Footprint};

#[derive(Debug, Clone, PartialEq)]
pub enum RoofShape {
    Flat,
    Gabled,
    Hipped,
    Pyramidal,
    Skillion,
    Mansard,
    Unknown(String),
}

impl RoofShape {
    fn parse(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "flat" => Self::Flat,
            "gabled" | "gable" => Self::Gabled,
            "hipped" | "hip" => Self::Hipped,
            "pyramidal" | "pyramid" => Self::Pyramidal,
            "skillion" | "shed" | "lean-to" => Self::Skillion,
            "mansard" => Self::Mansard,
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn suggested_max_planes(&self) -> usize {
        match self {
            Self::Flat => 1,
            Self::Skillion => 1,
            Self::Gabled => 2,
            Self::Hipped | Self::Pyramidal => 4,
            Self::Mansard => 12,
            Self::Unknown(_) => 30,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RoofOrientation {
    Across,
    Along,
}

#[derive(Debug, Clone)]
pub struct BuildingHint {
    pub total_height: Option<f64>,
    pub roof_shape: Option<RoofShape>,
    pub roof_direction: Option<f64>,
    pub roof_orientation: Option<RoofOrientation>,
    pub roof_height: Option<f64>,
    pub num_floors: Option<i64>,
}

impl BuildingHint {
    pub fn from_footprint(fp: &Footprint) -> Self {
        let attrs = &fp.attributes.0;

        let total_height = match attrs.get("height") {
            Some(AttributeValue::Float(v)) if *v > 0.0 => Some(*v),
            Some(AttributeValue::Int(v)) if *v > 0 => Some(*v as f64),
            _ => None,
        };

        let num_floors = match attrs.get("num_floors") {
            Some(AttributeValue::Int(v)) if *v > 0 => Some(*v),
            _ => None,
        };

        let roof_shape = match attrs.get("roof_shape") {
            Some(AttributeValue::String(s)) if !s.is_empty() => Some(RoofShape::parse(s)),
            _ => None,
        };

        let roof_direction = match attrs.get("roof_direction") {
            Some(AttributeValue::Float(v)) => Some(*v),
            Some(AttributeValue::Int(v)) => Some(*v as f64),
            _ => None,
        };

        let roof_orientation = match attrs.get("roof_orientation") {
            Some(AttributeValue::String(s)) => match s.to_lowercase().trim() {
                "across" => Some(RoofOrientation::Across),
                "along" => Some(RoofOrientation::Along),
                _ => None,
            },
            _ => None,
        };

        let roof_height = match attrs.get("roof_height") {
            Some(AttributeValue::Float(v)) if *v > 0.0 => Some(*v),
            Some(AttributeValue::Int(v)) if *v > 0 => Some(*v as f64),
            _ => None,
        };

        Self {
            total_height,
            roof_shape,
            roof_direction,
            roof_orientation,
            roof_height,
            num_floors,
        }
    }

    pub fn estimated_height(&self) -> Option<f64> {
        self.total_height
            .or_else(|| self.num_floors.map(|n| n as f64 * 3.0))
    }

    pub fn h_eave(&self, h_ground: f64) -> Option<f64> {
        let h_total = self.total_height?;
        let h_roof = self.roof_height?;
        if h_roof < h_total {
            Some(h_ground + h_total - h_roof)
        } else {
            None
        }
    }

    pub fn best_roof_height(&self, z_70p: f64, h_ground: f64) -> f64 {
        if let Some(h) = self.estimated_height() {
            h_ground + h
        } else if z_70p > h_ground {
            z_70p
        } else {
            h_ground + 5.0
        }
    }

    pub fn is_flat(&self) -> bool {
        matches!(self.roof_shape, Some(RoofShape::Flat))
    }

    /// Z ceiling: reject lidar points above this height.
    pub fn z_ceiling(&self, h_ground: f64) -> Option<f64> {
        self.estimated_height().map(|h| h_ground + h + 1.0)
    }
}
