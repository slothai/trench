import "../../diku-dk/linalg/linalg"

-- https://futhark-lang.org/docs/doc/futlib/math.html
module type field = {
    type t
    val +: t -> t -> t
    val -: t -> t -> t
    val *: t -> t -> t
    val /: t -> t -> t
    val i32: i32 -> t
}

-- Usually weight_t would be f32 or f64
module type weight_t = {
    type t
    val i32: i32 -> t
}

module type layers_t = {
    type t
    val dense: i32 -> i32
}

module layers (T: weight_t): layers_t with t = T.t = {
    type t = T.t
    let dense (size: i32) = size
}

module type trench_t = {
    type t
    module layers: layers_t
}

module trench (T: weight_t): trench_t with t = T.t = {
    type t = T.t
    module layers = layers T
}

type^ forwards 'input 'w 'output 'cache = w -> input -> (output, cache)
-- type backwards  'c 'w  'err_in  'err_out '^u = bool -> u -> w -> c -> err_in  -> (err_out, w)

type^ NN 'input 'w 'output 'cache = { forward : forwards input w output cache, weights : w}

type^ activation_func 'o = {f:o -> o, prime:o -> o}

module Dense(T: field) = {
    type t = T.t

    module linalg   = mk_linalg T

    let forward (m: i32) (nin: i32) (nout: i32)
                (w: [nin][nout]t) (b: [nout]t)
                (input: [m][nin]t) =
        let res = linalg.matmul input w
        let res_with_bias = map (\x -> map2 (T.+) x b) res
        in res_with_bias

    let init nin nout =
        let w = replicate nin (replicate nout (T.i32 0))
        let b = replicate nout (T.i32 0)
        in
        {
            forward = \m -> forward m nin nout w b,
            weights = (w, b)
        } 
}