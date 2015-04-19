
/**
 * Module dependencies.
 */

var benchmark = require('benchmark')
  , colors = require('colors')
  , io = require('../')
  , parser = io.parser
  , suite = new benchmark.Suite('Decode packet');

suite.add('string', function () {
  parser.decodePacket('4:::"2"');
});

suite.add('event', function () {
  parser.decodePacket('5:::{"name":"woot"}');
});

suite.add('event+ack', function () {
  parser.decodePacket('5:1+::{"name":"tobi"}');
});

suite.add('event+data', function () {
  parser.decodePacket('5:::{"name":"edwald","args":[{"a": "b"},2,"3"]}');
});

suite.add('heartbeat', function () {
  parser.decodePacket('2:::');
});

suite.add('error', function () {
  parser.decodePacket('7:::2+0');
});

var payload = parser.encodePayload([
    parser.encodePacket({ type: 'message', data: '5', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: '53d', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: 'foobar', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: 'foobarbaz', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: 'foobarbazfoobarbaz', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: 'foobarbaz', endpoint: '' })
  , parser.encodePacket({ type: 'message', data: 'foobar', endpoint: '' })
]);

suite.add('payload', function () {
  parser.decodePayload(payload);
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
