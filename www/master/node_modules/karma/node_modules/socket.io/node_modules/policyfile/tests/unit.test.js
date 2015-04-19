var fspfs = require('../')
  , fs = require('fs')
  , http = require('http')
  , https = require('https')
  , net = require('net')
  , should = require('should')
  , assert = require('assert');

module.exports = {
  // Library version should be Semver compatible
  'Library version': function(){
     fspfs.version.should.match(/^\d+\.\d+\.\d+$/);
  }

  // Creating a server instace should not cause any problems
  // either using the new Server or createServer method.
, 'Create Server instance': function(){
    var server = fspfs.createServer()
      , server2 = new fspfs.Server({log:false}, ['blog.3rd-Eden.com:1337']);
    
    // server 2 options test
    server2.log.should.be.false;
    server2.origins.length.should.equal(1);
    server2.origins[0].should.equal('blog.3rd-Eden.com:1337');
    
    // server defaults
    (typeof server.log).should.be.equal('function');
    server.origins.length.should.equal(1);
    server.origins[0].should.equal('*:*');
    
    // instance checking, sanity check
    assert.ok(server instanceof fspfs.Server);
    assert.ok(!!server.buffer);
    
    // more options testing
    server = fspfs.createServer(['blog.3rd-Eden.com:80']);
    server.origins.length.should.equal(1);
    server.origins[0].should.equal('blog.3rd-Eden.com:80');
    
    server = fspfs.createServer({log:false},['blog.3rd-Eden.com:80']);
    server.log.should.be.false;
    server.origins.length.should.equal(1);
    server.origins[0].should.equal('blog.3rd-Eden.com:80');
    
  }

, 'Add origin': function(){
    var server = fspfs.createServer();
    server.add('google.com:80', 'blog.3rd-Eden.com:1337');
    
    server.origins.length.should.equal(3);
    server.origins.indexOf('google.com:80').should.be.above(0);
    
    // don't allow duplicates
    server.add('google.com:80', 'google.com:80');
    
    var i = server.origins.length
      , count = 0;
    
    while(i--){
      if (server.origins[i] === 'google.com:80'){
        count++;
      }
    }
    
    count.should.equal(1);
  }

, 'Remove origin': function(){
    var server = fspfs.createServer();
    server.add('google.com:80', 'blog.3rd-Eden.com:1337');
    server.origins.length.should.equal(3);
    
    server.remove('google.com:80');
    server.origins.length.should.equal(2);
    server.origins.indexOf('google.com:80').should.equal(-1);
  }

, 'Buffer': function(){
    var server = fspfs.createServer();
    
    Buffer.isBuffer(server.buffer).should.be.true;
    server.buffer.toString().indexOf('to-ports="*"').should.be.above(0);
    server.buffer.toString().indexOf('domain="*"').should.be.above(0);
    server.buffer.toString().indexOf('domain="google.com"').should.equal(-1);
    
    // The buffers should be rebuild when new origins are added
    server.add('google.com:80');
    server.buffer.toString().indexOf('to-ports="80"').should.be.above(0);
    server.buffer.toString().indexOf('domain="google.com"').should.be.above(0);
    
    server.remove('google.com:80');
    server.buffer.toString().indexOf('to-ports="80"').should.equal(-1);
    server.buffer.toString().indexOf('domain="google.com"').should.equal(-1);
  }

, 'Responder': function(){
    var server = fspfs.createServer()
      , calls = 0
      // dummy socket to emulate a `real` socket
      , dummySocket = {
          readyState: 'open'
        , end: function(buffer){
          calls++;
          Buffer.isBuffer(buffer).should.be.true;
          buffer.toString().should.equal(server.buffer.toString());
        }
      };
    
    server.responder(dummySocket);
    calls.should.equal(1);
  }

, 'Event proxy': function(){
    var server = fspfs.createServer()
      , calls = 0;
    
    Object.keys(process.EventEmitter.prototype).forEach(function proxy(key){
      assert.ok(!!server[key] && typeof server[key] === 'function');
    });
    
    // test if it works by calling a none default event
    server.on('pew', function(){
      calls++;
    });
    
    server.emit('pew');
    calls.should.equal(1);
  }

, 'inline response http': function(){
    var port = 1335
      , httpserver = http.createServer(function(q,r){r.writeHead(200);r.end(':3')})
      , server = fspfs.createServer();
    
    httpserver.listen(port, function(){
      server.listen(port + 1, httpserver, function(){
        var client = net.createConnection(port);
        client.write('<policy-file-request/>\0');
        client.on('error', function(err){
          assert.ok(!err, err)
        });
        client.on('data', function(data){
        
          var response = data.toString();
          console.log(response);
          
          response.indexOf('to-ports="*"').should.be.above(0);
          response.indexOf('domain="*"').should.be.above(0);
          response.indexOf('domain="google.com"').should.equal(-1);
          
          // clean up
          client.destroy();
          server.close();
          httpserver.close();
        });
      });
    });
  }

, 'server response': function(){
    var port = 1340
      , server = fspfs.createServer();
      
    server.listen(port, function(){
      var client = net.createConnection(port);
      client.write('<policy-file-request/>\0');
      client.on('error', function(err){
        assert.ok(!err, err)
      });
      client.on('data', function(data){
      
        var response = data.toString();
        
        response.indexOf('to-ports="*"').should.be.above(0);
        response.indexOf('domain="*"').should.be.above(0);
        response.indexOf('domain="google.com"').should.equal(-1);
        
        // clean up
        client.destroy();
        server.close();
      });
    });
  }

, 'inline response https': function(){
    var port = 1345
      , ssl = {
          key: fs.readFileSync(__dirname + '/ssl/ssl.private.key').toString()
        , cert: fs.readFileSync(__dirname + '/ssl/ssl.crt').toString()
        }
      , httpserver = https.createServer(ssl, function(q,r){r.writeHead(200);r.end(':3')})
      , server = fspfs.createServer();
    
    httpserver.listen(port, function(){
      server.listen(port + 1, httpserver, function(){
        var client = net.createConnection(port);
        client.write('<policy-file-request/>\0');
        client.on('error', function(err){
          assert.ok(!err, err)
        });
        client.on('data', function(data){
        
          var response = data.toString();
          
          response.indexOf('to-ports="*"').should.be.above(0);
          response.indexOf('domain="*"').should.be.above(0);
          response.indexOf('domain="google.com"').should.equal(-1);
          
          // clean up
          client.destroy();
          server.close();
          httpserver.close();
        });
      });
    });
  }

, 'connect_failed': function(){
    var server = fspfs.createServer();
    
    server.on('connect_failed', function(){
      assert.ok(true);
    });
    
    server.listen(function(){
      assert.ok(false, 'Run this test without root access');
      server.close();
    });
  }
};