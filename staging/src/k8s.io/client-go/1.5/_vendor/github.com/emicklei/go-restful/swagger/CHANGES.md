Change history of swagger
=
2015-10-16
- add type override mechanism for swagger models (MR 254, nathanejohnson)
- replace uses of wildcard in generated apidocs (issue 251)

2015-05-25
- (api break) changed the type of Properties in Model
- (api break) changed the type of Models in ApiDeclaration
- (api break) changed the parameter type of PostBuildDeclarationMapFunc

2015-04-09
- add ModelBuildable interface for customization of Model

2015-03-17
- preserve order of Routes per WebService in Swagger listing
- fix use of $ref and type in Swagger models
- add api version to listing

2014-11-14
- operation parameters are now sorted using ordering path,query,form,header,body

2014-11-12
- respect omitempty tag value for embedded structs
- expose ApiVersion of WebService to Swagger ApiDeclaration

2014-05-29
- (api add) Ability to define custom http.Handler to serve swagger-ui static files

2014-05-04
- (fix) include model for array element type of response

2014-01-03
- (fix) do not add primitive type to the Api models

2013-11-27
- (fix) make Swagger work for WebServices with root ("/" or "") paths

2013-10-29
- (api add) package variable LogInfo to customize logging function

2013-10-15
- upgraded to spec version 1.2 (https://github.com/wordnik/swagger-core/wiki/1.2-transition)