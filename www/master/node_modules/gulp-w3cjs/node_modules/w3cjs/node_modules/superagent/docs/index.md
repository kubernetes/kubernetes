# SuperAgent

 Super Agent is light-weight progressive ajax API crafted for flexibility, readability, and a low learning curve after being frustrated with many of the existing request APIs.

     request
       .post('/api/pet')
       .data({ name: 'Manny', species: 'cat' })
       .set('X-API-Key', 'foobar')
       .set('Accept', 'application/json')
       .end(function(res){
         if (res.ok) {
           alert('yay got ' + JSON.stringify(res.body));
         } else {
           alert('Oh no! error ' + res.text);
         }
       });

## Request basics

 A request can be initiated by invoking the appropriate method on the `request` object, then calling `.end()` to send the request. For example a simple GET request:
 
     request
       .get('/search')
       .end(function(res){
       
       });

 The __node__ client may also provide absolute urls:

     request
       .get('http://example.com/search')
       .end(function(res){
     
       });

  __DELETE__, __HEAD__, __POST__, __PUT__ and other __HTTP__ verbs may also be used, simply change the method name:
  
    request
      .head('/favicon.ico')
      .end(function(res){
      
      });

  __DELETE__ is a special-case, as it's a reserved word, so the method is named `.del()`:
  
    request
      .del('/user/1')
      .end(function(res){
        
      });

### Crafting requests

  SuperAgent's flexible API gives you the granularity you need, _when_ you need, yet more concise variations help reduce the amount of code necessary. For example the following GET request:
  
    request
      .get('/search')
      .end(function(res){
    
      });

  Could also be defined as the following, where a callback is given to the HTTP verb method:
  
    request
      .get('/search', function(res){
    
      });

   Taking this further the default HTTP verb is __GET__ so the following works as well:
   
     request('/search', function(res){
 
     });

   This applies to more complicated requests as well, for example the following __GET__ request with a query-string can be written in the chaining manner:
   
     request
       .get('/search')
       .data({ query: 'tobi the ferret' })
       .end(function(res){
         
       });

   Or one may pass the query-string object to `.get()`:
   
     request
       .get('/search', { query: 'tobi the ferret' })
       .end(function(res){
       
       });
  
  Taking this even further the callback may be passed as well:
  
     request
       .get('/search', { query: 'tobi the ferret' }, function(res){
     
       });

## Dealing with errors

  On a network error (e.g. connection refused or timeout), SuperAgent emits
  `error` unless you pass `.end()` a callback with two parameters. Then
  SuperAgent will invoke it with the error first, followed by a null response.

     request
       .get('http://wrongurl')
       .end(function(err, res){
         console.log('ERROR: ', err)
       });

  On HTTP errors instead, SuperAgent populates the response with flags
  indicating the error. See `Response status` below.

## Setting header fields

  Setting header fields is simple, invoke `.set()` with a field name and value:
  
     request
       .get('/search')
       .set('API-Key', 'foobar')
       .set('Accept', 'application/json')
       .end(callback);

## GET requests

 The `.data()` method accepts objects, which when used with the __GET__ method will form a query-string. The following will produce the path `/search?query=Manny&range=1..5&order=desc`.
 
     request
       .get('/search')
       .data({ query: 'Manny' })
       .data({ range: '1..5' })
       .data({ order: 'desc' })
       .end(function(res){

       });

  The `.data()` method accepts strings as well:
  
      request
        .get('/querystring')
        .data('search=Manny&range=1..5')
        .end(function(res){

        });

### POST / PUT requests

  A typical JSON __POST__ request might look a little like the following, where we set the Content-Type header field appropriately, and "write" some data, in this case just a JSON string.

      request.post('/user')
        .set('Content-Type', 'application/json')
        .data('{"name":"tj","pet":"tobi"})
        .end(callback)

  Since JSON is undoubtably the most common, it's the _default_! The following example is equivalent to the previous.

      request.post('/user')
        .data({ name: 'tj', pet: 'tobi' })
        .end(callback)

  Or using multiple `.data()` calls:
  
      request.post('/user')
        .data({ name: 'tj' })
        .data({ pet: 'tobi' })
        .end(callback)

  SuperAgent formats are extensible, however by default "json" and "form" are supported. To send the data as `application/x-www-form-urlencoded` simply invoke `.type()` with "form-data", where the default is "json". This request will POST the body "name=tj&pet=tobi".

      request.post('/user')
        .type('form')
        .data({ name: 'tj' })
        .data({ pet: 'tobi' })
        .end(callback)

## Response properties

  Many helpful flags and properties are set on the `Response` object, ranging from the response text, parsed response body, header fields, status flags and more.
  
### Response text

  The `res.text` property contains the unparsed response body string.

### Response body

  Much like SuperAgent can auto-serialize request data, it can also automatically parse it. When a parser is defined for the Content-Type, it is parsed, which by default includes "application/json" and "application/x-www-form-urlencoded". The parsed object is then available via `res.body`.

### Response header fields

  The `res.header` contains an object of parsed header fields, lowercasing field names much like node does. For example `res.header['content-length']`.

### Response Content-Type

  The Content-Type response header is special-cased, providing `res.contentType`, which is void of the charset (if any). For example the Content-Type of "text/html; charset=utf8" will provide "text/html" as `res.contentType`, and the `res.charset` property would then contain "utf8".

### Response status

  The response status flags help determine if the request was a success, among other useful information, making SuperAgent ideal for interacting with RESTful web services. These flags are currently defined as:
  
     var type = status / 100 | 0;

     // status / class
     res.status = status;
     res.statusType = type;

     // basics
     res.info = 1 == type;
     res.ok = 2 == type;
     res.clientError = 4 == type;
     res.serverError = 5 == type;
     res.error = 4 == type || 5 == type;

     // sugar
     res.accepted = 202 == status;
     res.noContent = 204 == status || 1223 == status;
     res.badRequest = 400 == status;
     res.unauthorized = 401 == status;
     res.notAcceptable = 406 == status;
     res.notFound = 404 == status;
