package services

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

object WinePredictor {
  def runPythonCode(parameters: List[Float]): Float = {
    val sys = py.module("sys")
    sys.path.append("app/services/python")
    val predictor = py.module("predictor")
    val valueInScala = predictor.run(parameters.toPythonProxy).as[Float]
    return valueInScala
  }
}