var Store = require('../helpers/store')
  , http = require('http')

//
// jsonp is a handy technique for getting around the limitations of the same-origin policy. 
// (http://en.wikipedia.org/wiki/Same_origin_policy) 
// 
// normally, to dynamically update a page you use an XmlHttpRequest. this has flakey support 
// is some browsers and is restricted by the same origin policy. you cannot perform XHR requests to
// someone else's server. one way around this would be to proxy requests to all the servers you want
// to xhr to, and your core server - so that everything has the same port and host.
// 
// another way, is to turn json into javascript. (which is exempt from the same origin policy) 
// this is done by wrapping the json object in a function call, and then including a script tag.
//
// here we're proxing our own JSON returning server, but we could proxy any server on the internet,
// and our client side app would be slurping down JSONP from anywhere.
// 
// curl localhost:1337/whatever?callback=alert
// alert([]) //which is valid javascript!
//
// also see http://en.wikipedia.org/wiki/JSONP#JSONP
//

http.createServer(new Store().handler()).listen(7531)

require('../../lib/node-http-proxy').createServer(
  require('connect-jsonp')(true),
  'localhost', 7531
).listen(1337)
