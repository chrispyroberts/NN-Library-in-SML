val xs = List.map (toValues) [[2.0, 3.0, ~1.0],
                              [3.0, ~1.0, 0.5],
                              [0.5, 1.0, 1.0],
                              [1.0, 1.0, ~1.0]]


val ys = toValues [1.0, ~1.0, ~1.0, 1.0] (* desired targets *) 

val randomCount = ref 10

val model = mkMLP(3, [4, 4, 1])
val epochs = 20
val lr = 0.1

(* train the model! *)
val (preds, stats) = train(model, xs, ys, epochs, lr)

