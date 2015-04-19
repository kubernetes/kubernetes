var assert = require('assert');

var uuid = require('../');

// Verify ordering of v1 ids created with explicit times
var TIME = 1321644961388; // 2011-11-18 11:36:01.388-08:00

function compare(name, ids) {
  test(name, function() {
    // avoid .map for older browsers
    for (var i=0 ; i<ids.length ; ++i) {
      ids[i] = ids[i].split('-').reverse().join('-');
    }
    ids = ids.sort();
    var sorted = ([].concat(ids)).sort();

    assert(sorted.toString() == ids.toString(), name + ' have expected order');
  });
}

// Verify ordering of v1 ids created using default behavior
compare('uuids with current time', [
  uuid.v1(),
  uuid.v1(),
  uuid.v1(),
  uuid.v1(),
  uuid.v1()
]);

// Verify ordering of v1 ids created with explicit times
compare('uuids with time option', [
  uuid.v1({msecs: TIME - 10*3600*1000}),
  uuid.v1({msecs: TIME - 1}),
  uuid.v1({msecs: TIME}),
  uuid.v1({msecs: TIME + 1}),
  uuid.v1({msecs: TIME + 28*24*3600*1000})
]);

test('msec', function() {
  assert(
    uuid.v1({msecs: TIME}) != uuid.v1({msecs: TIME}),
    'IDs created at same msec are different'
  );
});

test('exception thrown when > 10k ids created in 1ms', function() {
  // Verify throw if too many ids created
  var thrown = false;
  try {
    uuid.v1({msecs: TIME, nsecs: 10000});
  } catch (e) {
    thrown = true;
  }
  assert(thrown, 'Exception thrown when > 10K ids created in 1 ms');
});

test('clock regression by msec', function() {
  // Verify clock regression bumps clockseq
  var uidt = uuid.v1({msecs: TIME});
  var uidtb = uuid.v1({msecs: TIME - 1});
  assert(
    parseInt(uidtb.split('-')[3], 16) - parseInt(uidt.split('-')[3], 16) === 1,
    'Clock regression by msec increments the clockseq'
  );
});

test('clock regression by nsec', function() {
  // Verify clock regression bumps clockseq
  var uidtn = uuid.v1({msecs: TIME, nsecs: 10});
  var uidtnb = uuid.v1({msecs: TIME, nsecs: 9});
  assert(
    parseInt(uidtnb.split('-')[3], 16) - parseInt(uidtn.split('-')[3], 16) === 1,
    'Clock regression by nsec increments the clockseq'
  );
});

test('explicit options product expected id', function() {
  // Verify explicit options produce expected id
  var id = uuid.v1({
    msecs: 1321651533573,
    nsecs: 5432,
    clockseq: 0x385c,
    node: [ 0x61, 0xcd, 0x3c, 0xbb, 0x32, 0x10 ]
  });
  assert(id == 'd9428888-122b-11e1-b85c-61cd3cbb3210', 'Explicit options produce expected id');
});

test('ids spanning 1ms boundary are 100ns apart', function() {
  // Verify adjacent ids across a msec boundary are 1 time unit apart
  var u0 = uuid.v1({msecs: TIME, nsecs: 9999});
  var u1 = uuid.v1({msecs: TIME + 1, nsecs: 0});

  var before = u0.split('-')[0], after = u1.split('-')[0];
  var dt = parseInt(after, 16) - parseInt(before, 16);
  assert(dt === 1, 'Ids spanning 1ms boundary are 100ns apart');
});

test('parse/unparse', function() {
  var id = '00112233445566778899aabbccddeeff';
  assert(uuid.unparse(uuid.parse(id.substr(0,10))) ==
    '00112233-4400-0000-0000-000000000000', 'Short parse');
  assert(uuid.unparse(uuid.parse('(this is the uuid -> ' + id + id)) ==
    '00112233-4455-6677-8899-aabbccddeeff', 'Dirty parse');
});

