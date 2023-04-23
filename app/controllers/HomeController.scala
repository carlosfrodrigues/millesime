package controllers

import javax.inject.Inject
import services.WinePredictor
import play.api.data._
import play.api.i18n._
import play.api.mvc._

class HomeController @Inject()(cc: MessagesControllerComponents) extends MessagesAbstractController(cc) {
  def index = Action {
    Ok(views.html.index())
  }

  def processForm = Action { request =>
    val postVals = request.body.asFormUrlEncoded
    val parameters = postVals.map(_.values.flatten.map(_.toFloat).toList).getOrElse(List.empty[Float])
    val response = WinePredictor.runPythonCode(parameters)
    Ok(views.html.result(s"$response"))
  }
}
