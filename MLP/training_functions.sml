(* loss : value list * value list -> value *)
fun loss ([], []) = mkDefaultValue(0.0)
  | loss (pred::xs, y::ys) = add(mul(subtract(pred, y), subtract(pred, y)), loss(xs, ys))

(* train : mpl * value list list * value list * int * real ->  real list * unit list *)
fun train(model, data, target, numEpochs, lr) =
  let
    fun flatten l = foldl op@ [] l
    
    val parameters = getMLPParams(model)
    fun resetGradients () = foldl (fn (n, ()) => ((getGrad n) := 0.0)) () parameters
    
    fun updateParams() : unit = 
      let
        fun update(n : value) : unit = (getData n := !(getData n) - lr * !(getGrad n))
      in
        List.foldl (fn (n, ()) => update(n)) () parameters
      end
        
    fun trainHelp(model, 0, currEpoch, final_prediction, epochStatList) = (toList final_prediction, epochStatList)
      | trainHelp(model, n, currEpoch, _, epochStatList) =
      let
        val y_pred = flatten (List.map (fn x => forwardMLP(model, x)) data)
        val l = loss(y_pred, target)
        
        (* reset gradients of all parameters to 0 before calling backwards pass*)
        val () = resetGradients ()
        
        (* update gradients *)
        val () = backwards(l) 

        (* update params *)
        val () = updateParams ()

        (* Save stats *)
        val stats = (fn () => print("Epoch: " ^ Int.toString currEpoch ^ 
                                   " Loss: " ^ (Real.toString (!(getData l))) 
                                   ^ "\n"))
      in
        trainHelp(model, n-1, currEpoch+1, y_pred, stats :: epochStatList)
      end
  in
    trainHelp(model, numEpochs, 0, [], [])
  end