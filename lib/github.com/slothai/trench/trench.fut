import "../../diku-dk/linalg/linalg"
import "../../diku-dk/cpprandom/random"

-- https://futhark-lang.org/docs/doc/futlib/math.html
module type field = {
    type t
    val +: t -> t -> t
    val -: t -> t -> t
    val *: t -> t -> t
    val /: t -> t -> t
    val i32: i32 -> t
    val f32: f32 -> t
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

module Init(R: real) = {
    type t = R.t

    module uni = uniform_real_distribution R minstd_rand
    module norm = normal_distribution R minstd_rand

    -- See: https://futhark-lang.org/docs/doc/futlib/random.html#examples

    -- Generate 1-d array of uniformly distributed numbers
    let uni1 (seed: []i32) (range: (R.t, R.t)) (d1: i32): [d1]t =
        let rng = minstd_rand.rng_from_seed seed
        let rngs = minstd_rand.split_rng d1 rng
        let (_, xs) = unzip (map (\rng -> uni.rand range rng) rngs)
        in xs

    let norm1 (seed: []i32) (params: {mean:R.t, stddev:R.t}) (d1: i32): [d1]t =
        let rng = minstd_rand.rng_from_seed seed
        let rngs = minstd_rand.split_rng d1 rng
        let (_, xs) = unzip (map (\rng -> norm.rand params rng) rngs)
        in xs

    -- For 2/3/4-d arrays it's possible to generate a random 1-d array and unflatten it
    -- Array functions: https://futhark-lang.org/docs/doc/futlib/array.html
    -- Comment from the author: https://github.com/diku-dk/futhark/issues/520

    -- Initialization He and Glorot: https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init

    let glorot_uniform (seed: []i32) (d1: i32) (d2: i32): [d1][d2]t =
        let limit = R.(sqrt((i32 6) / (i32 d1 + i32 d2)))
        let range = (R.(negate limit), limit)
        in unflatten d1 d2 (uni1 seed range (d1*d2))

    let glorot_norm (seed: []i32) (d1: i32) (d2: i32): [d1][d2]t =
        let std = R.(sqrt((i32 2) / (i32 d1 + i32 d2)))
        in unflatten d1 d2 (norm1 seed {mean=(R.i32 0), stddev=std} (d1*d2))

    let he_uiform (seed: []i32) (d1: i32) (d2: i32): [d1][d2]t =
        let limit = R.(sqrt((i32 12) / (i32 d1 + i32 d2)))
        let range = (R.(negate limit), limit)
        in unflatten d1 d2 (uni1 seed range (d1*d2))

    let he_norm (seed: []i32) (d1: i32) (d2: i32): [d1][d2]t =
        let std = R.(sqrt((i32 4) / (i32 d1 + i32 d2)))
        in unflatten d1 d2 (norm1 seed {mean=(R.i32 0), stddev=std} (d1*d2))

    let zeros (d1: i32): [d1]t = replicate d1 (R.i32 0)
}

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