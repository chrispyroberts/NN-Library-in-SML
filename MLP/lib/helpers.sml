(* for making random functions *)
val randomCount = ref 1

(* returns a random real between -1 and 1 *)
fun uniform () =
    let
      val r = (randomCount := !randomCount+1; Random.rand(0, !randomCount))
      val sign = Random.randRange (0, 1) r
      val rand = Random.randReal r
    in
      case sign of
           0 => rand * ~1.0
         | _ => rand
    end

(* A dot product sum operator for values *)
fun dot ([], []) = (mkDefaultValue 0.0)
  | dot (w::ws, x::xs) = add(mul(w, x), dot (ws, xs))
  | dot (_, _) = raise IncompatibleSizes

(* function to convert some list of reals to values *)
fun toValues [] = []
  | toValues (x::xs) = (mkDefaultValue x) :: (toValues xs)

(* zip helper! *)
fun zip ([], []) = []
  | zip (x::xs, y::ys) = (x, y) :: zip(xs, ys)


(* helper! *)
(* value list -> real list *)
fun toList vs = List.map (fn v => !(getData v)) vs
