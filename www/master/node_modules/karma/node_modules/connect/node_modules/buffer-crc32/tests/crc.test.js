var crc32 = require('..');
var test = require('tap').test;

test('simple crc32 is no problem', function (t) {
  var input = Buffer('hey sup bros');
  var expected = Buffer([0x47, 0xfa, 0x55, 0x70]);
  t.same(crc32(input), expected);
  t.end();
});

test('another simple one', function (t) {
  var input = Buffer('IEND');
  var expected = Buffer([0xae, 0x42, 0x60, 0x82]);
  t.same(crc32(input), expected);
  t.end();
});

test('slightly more complex', function (t) {
  var input = Buffer([0x00, 0x00, 0x00]);
  var expected = Buffer([0xff, 0x41, 0xd9, 0x12]);
  t.same(crc32(input), expected);
  t.end();
});

test('complex crc32 gets calculated like a champ', function (t) {
  var input = Buffer('शीर्षक');
  var expected = Buffer([0x17, 0xb8, 0xaf, 0xf1]);
  t.same(crc32(input), expected);
  t.end();
});

test('casts to buffer if necessary', function (t) {
  var input = 'शीर्षक';
  var expected = Buffer([0x17, 0xb8, 0xaf, 0xf1]);
  t.same(crc32(input), expected);
  t.end();
});

test('can do signed', function (t) {
  var input = 'ham sandwich';
  var expected = -1891873021;
  t.same(crc32.signed(input), expected);
  t.end();
});

test('can do unsigned', function (t) {
  var input = 'bear sandwich';
  var expected = 3711466352;
  t.same(crc32.unsigned(input), expected);
  t.end();
});


test('simple crc32 in append mode', function (t) {
  var input = [Buffer('hey'), Buffer(' '), Buffer('sup'), Buffer(' '), Buffer('bros')];
  var expected = Buffer([0x47, 0xfa, 0x55, 0x70]);
  for (var crc = 0, i = 0; i < input.length; i++) {
    crc = crc32(input[i], crc);
  }
  t.same(crc, expected);
  t.end();
});


test('can do signed in append mode', function (t) {
  var input1 = 'ham';
  var input2 = ' ';
  var input3 = 'sandwich';
  var expected = -1891873021;

  var crc = crc32.signed(input1);
  crc = crc32.signed(input2, crc);
  crc = crc32.signed(input3, crc);

  t.same(crc, expected);
  t.end();
});

test('can do unsigned in append mode', function (t) {
  var input1 = 'bear san';
  var input2 = 'dwich';
  var expected = 3711466352;

  var crc = crc32.unsigned(input1);
  crc = crc32.unsigned(input2, crc);
  t.same(crc, expected);
  t.end();
});

