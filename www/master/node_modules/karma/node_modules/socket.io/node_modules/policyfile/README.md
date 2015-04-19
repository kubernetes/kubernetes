## LOL, WUT?
It basically allows you to allow or disallow Flash Player sockets from accessing your site.

## Installation

```bash
npm install policyfile
```
## Usage

The server is based on the regular and know `net` and `http` server patterns. So it you can just listen
for all the events that a `net` based server emits etc. But there is one extra event, the `connect_failed`
event. This event is triggered when we are unable to listen on the supplied port number.

### createServer
Creates a new server instance and accepts 2 optional arguments:

-  `options` **Object** Options to configure the server instance
    -  `log` **Boolean** Enable logging to STDOUT and STDERR (defaults to true)
-  `origins` **Array** An Array of origins that are allowed by the server (defaults to *:*)

```js
var pf = require('policyfile');
pf.createServer();
pf.listen();
```

#### server.listen
Start listening on the server and it takes 3 optional arguments

-  `port` **Number** On which port number should we listen? (defaults to 843, which is the first port number the FlashPlayer checks)
-  `server` **Server** A http server, if we are unable to accept requests or run the server we can also answer the policy requests inline over the supplied HTTP server.
-  `callback` **Function** A callback function that is called when listening to the server was successful.

```js
var pf = require('policyfile');
pf.createServer();
pf.listen(1337, function(){
  console.log(':3 yay')
});
```

Changing port numbers can be handy if you do not want to run your server as root and have port 843 forward to a non root port number (aka a number above 1024).

```js
var pf = require('policyfile')
  , http = require('http');

server = http.createServer(function(q,r){r.writeHead(200);r.end('hello world')});
server.listen(80);

pf.createServer();
pf.listen(1337, server, function(){
  console.log(':3 yay')
});
```

Support for serving inline requests over a existing HTTP connection as the FlashPlayer will first check port 843, but if it's unable to get a response there it will send a policy file request over port 80, which is usually your http server.

#### server.add
Adds more origins to the policy file you can add as many arguments as you like.

```js
var pf = require('policyfile');
pf.createServer(['google.com:80']);
pf.listen();
pf.add('blog.3rd-Eden.com:80', 'blog.3rd-Eden.com:8080'); // now has 3 origins
```

#### server.add
Adds more origins to the policy file you can add as many arguments as you like.

```js
var pf = require('policyfile');
pf.createServer(['blog.3rd-Eden.com:80', 'blog.3rd-Eden.com:8080']);
pf.listen();
pf.remove('blog.3rd-Eden.com:8080'); // only contains the :80 version now
```

#### server.close
Shuts down the server

```js
var pf = require('policyfile');
pf.createServer();
pf.listen();
pf.close(); // OH NVM.
```

## API
http://3rd-eden.com/FlashPolicyFileServer/

## Examples
See https://github.com/3rd-Eden/FlashPolicyFileServer/tree/master/examples for examples

## Licence

MIT see LICENSE file in the repository