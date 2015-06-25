How to use Swagger UI with go-restful
=

Get the Swagger UI sources (version 1.2 only)

	git clone https://github.com/wordnik/swagger-ui.git
	
The project contains a "dist" folder.
Its contents has all the Swagger UI files you need.

The `index.html` has an `url` set to `http://petstore.swagger.wordnik.com/api/api-docs`.
You need to change that to match your WebService JSON endpoint  e.g. `http://localhost:8080/apidocs.json`

Now, you can install the Swagger WebService for serving the Swagger specification in JSON.

	config := swagger.Config{
		WebServices:    restful.RegisteredWebServices(),
		ApiPath:        "/apidocs.json",
		SwaggerPath:     "/apidocs/",
		SwaggerFilePath: "/Users/emicklei/Projects/swagger-ui/dist"}
	swagger.InstallSwaggerService(config)		
	
	
Notes
--
- The Nickname of an Operation is automatically set by finding the name of the function. You can override it using RouteBuilder.Operation(..) 
- The WebServices field of swagger.Config can be used to control which service you want to expose and document ; you can have multiple configs and therefore multiple endpoints.
- Use tag "description" to annotate a struct field with a description to show in the UI