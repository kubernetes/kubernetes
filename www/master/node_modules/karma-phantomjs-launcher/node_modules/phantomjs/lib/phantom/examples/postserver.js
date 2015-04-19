// Example using HTTP POST operation

var page = require('webpage').create(),
    server = require('webserver').create(),
    system = require('system'),
    data = 'universe=expanding&answer=42';

if (system.args.length !== 2) {
    console.log('Usage: postserver.js <portnumber>');
    phantom.exit(1);
}

var port = system.args[1];

service = server.listen(port, function (request, response) {
    console.log('Request received at ' + new Date());

    response.statusCode = 200;
    response.headers = {
        'Cache': 'no-cache',
        'Content-Type': 'text/plain;charset=utf-8'
    };
    response.write(JSON.stringify(request, null, 4));
    response.close();
});

page.open('http://localhost:' + port + '/', 'post', data, function (status) {
    if (status !== 'success') {
        console.log('Unable to post!');
    } else {
        console.log(page.plainText);
    }
    phantom.exit();
});
