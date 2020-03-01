-- Usually this would be f32 or f64
module type weight_t = {
    type t
    val i32: i32 -> t
}

module type layers_t = {
    type t
    val zero: i32
}

module layers (T: weight_t): layers_t with t = T.t = {
    type t = T.t
    let zero: i32 = 0
}

module type trench_t = {
    type t
    module layers: layers_t
}

module trench (T: weight_t): trench_t with t = T.t = {
    type t = T.t
    module layers = layers T
}