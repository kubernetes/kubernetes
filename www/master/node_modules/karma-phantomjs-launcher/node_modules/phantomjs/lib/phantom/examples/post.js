// Example using HTTP POST operation

var page = require('webpage').create(),
    server = 'http://posttestserver.com/post.php?dump',
    data = 'universe=expanding&answer=42';

page.open(server, 'post', data, function (status) {
    if (status !== 'success') {
        console.log('Unable to post!');
    } else {
        console.log(page.content);
    }
    phantom.exit();
});
