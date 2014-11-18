Change history of swagger
=

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