# end-of-stream

A node module that calls a callback when a readable/writable/duplex stream has completed or failed.

	npm install end-of-stream

## Usage

Simply pass a stream and a callback to the `eos`.
Both legacy streams and streams2 are supported.

``` js
var eos = require('end-of-stream');

eos(readableStream, function(err) {
	if (err) return console.log('stream had an error or closed early');
	console.log('stream has ended');
});

eos(writableStream, function(err) {
	if (err) return console.log('stream had an error or closed early');
	console.log('stream has finished');
});

eos(duplexStream, function(err) {
	if (err) return console.log('stream had an error or closed early');
	console.log('stream has ended and finished');
});

eos(duplexStream, {readable:false}, function(err) {
	if (err) return console.log('stream had an error or closed early');
	console.log('stream has ended but might still be writable');
});

eos(duplexStream, {writable:false}, function(err) {
	if (err) return console.log('stream had an error or closed early');
	console.log('stream has ended but might still be readable');
});

eos(readableStream, {error:false}, function(err) {
	// do not treat emit('error', err) as a end-of-stream
});
```

## License

MIT