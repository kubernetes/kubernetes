var assert = require('assert');

var uuid = require('../');

var log = console.log;

var generators = {
  v1: uuid.v1,
  v4: uuid.v4
};

var UUID_FORMAT = {
  v1: /[0-9a-f]{8}-[0-9a-f]{4}-1[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/i,
  v4: /[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/i
};

var N = 1e4;

// Get %'age an actual value differs from the ideal value
function divergence(actual, ideal) {
  return Math.round(100*100*(actual - ideal)/ideal)/100;
}

function rate(msg, t) {
  log(msg + ': ' + (N / (Date.now() - t) * 1e3 | 0) + ' uuids\/second');
}

for (var version in generators) {
  var counts = {}, max = 0;
  var generator = generators[version];
  var format = UUID_FORMAT[version];

  log('\nSanity check ' + N + ' ' + version + ' uuids');
  for (var i = 0, ok = 0; i < N; i++) {
    id = generator();
    if (!format.test(id)) {
      throw Error(id + ' is not a valid UUID string');
    }

    if (id != uuid.unparse(uuid.parse(id))) {
      assert(fail, id + ' is not a valid id');
    }

    // Count digits for our randomness check
    if (version == 'v4') {
      var digits = id.replace(/-/g, '').split('');
      for (var j = digits.length-1; j >= 0; j--) {
        var c = digits[j];
        max = Math.max(max, counts[c] = (counts[c] || 0) + 1);
      }
    }
  }

  // Check randomness for v4 UUIDs
  if (version == 'v4') {
    // Limit that we get worried about randomness. (Purely empirical choice, this!)
    var limit = 2*100*Math.sqrt(1/N);

    log('\nChecking v4 randomness.  Distribution of Hex Digits (% deviation from ideal)');

    for (var i = 0; i < 16; i++) {
      var c = i.toString(16);
      var bar = '', n = counts[c], p = Math.round(n/max*100|0);

      // 1-3,5-8, and D-F: 1:16 odds over 30 digits
      var ideal = N*30/16;
      if (i == 4) {
        // 4: 1:1 odds on 1 digit, plus 1:16 odds on 30 digits
        ideal = N*(1 + 30/16);
      } else if (i >= 8 && i <= 11) {
        // 8-B: 1:4 odds on 1 digit, plus 1:16 odds on 30 digits
        ideal = N*(1/4 + 30/16);
      } else {
        // Otherwise: 1:16 odds on 30 digits
        ideal = N*30/16;
      }
      var d = divergence(n, ideal);

      // Draw bar using UTF squares (just for grins)
      var s = n/max*50 | 0;
      while (s--) bar += '=';

      assert(Math.abs(d) < limit, c + ' |' + bar + '| ' + counts[c] + ' (' + d + '% < ' + limit + '%)');
    }
  }
}

// Perf tests
for (var version in generators) {
  log('\nPerformance testing ' + version + ' UUIDs');
  var generator = generators[version];
  var buf = new uuid.BufferClass(16);

  for (var i = 0, t = Date.now(); i < N; i++) generator();
  rate('uuid.' + version + '()', t);

  for (var i = 0, t = Date.now(); i < N; i++) generator('binary');
  rate('uuid.' + version + '(\'binary\')', t);

  for (var i = 0, t = Date.now(); i < N; i++) generator('binary', buf);
  rate('uuid.' + version + '(\'binary\', buffer)', t);
}
