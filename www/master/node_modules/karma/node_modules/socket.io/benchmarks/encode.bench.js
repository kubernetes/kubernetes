
/**
 * Module dependencies.
 */

var benchmark = require('benchmark')
  , colors = require('colors')
  , io = require('../')
  , parser = io.parser
  , suite = new benchmark.Suite('Encode packet');

suite.add('string', function () {
  parser.encodePacket({
      type: 'json'
    , endpoint: ''
    , data: '2'
  });
});

suite.add('event', function () {
  parser.encodePacket({
      type: 'event'
    , name: 'woot'
    , endpoint: ''
    , args: []
  });
});

suite.add('event+ack', function () {
  parser.encodePacket({
      type: 'json'
    , id: 1
    , ack: 'data'
    , endpoint: ''
    , data: { a: 'b' }
  });
});

suite.add('event+data', function () {
  parser.encodePacket({
      type: 'event'
    , name: 'edwald'
    , endpoint: ''
    , args: [{a: 'b'}, 2, '3']
  });
});

suite.add('heartbeat', function () {
  parser.encodePacket({
      type: 'heartbeat'
    , endpoint: ''
  })
});

suite.add('error', function () {
  parser.encodePacket({
      type: 'error'
    , reason: 'unauthorized'
    , advice: 'reconnect'
    , endpoint: ''
  })
})

suite.add('payload', function () {
  parser.encodePayload([
      parser.encodePacket({ type: 'message', data: '5', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: '53d', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: 'foobar', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: 'foobarbaz', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: 'foobarbazfoobarbaz', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: 'foobarbaz', endpoint: '' })
    , parser.encodePacket({ type: 'message', data: 'foobar', endpoint: '' })
  ]);
});

suite.on('cycle', function (bench, details) {
  console.log('\n' + suite.name.grey, details.name.white.bold);
  console.log([
      details.hz.toFixed(2).cyan + ' ops/sec'.grey
    , details.count.toString().white + ' times executed'.grey
    , 'benchmark took '.grey + details.times.elapsed.toString().white + ' sec.'.grey
    , 
  ].join(', '.grey));
});

if (!module.parent) {
  suite.run();
} else {
  module.exports = suite;
}
