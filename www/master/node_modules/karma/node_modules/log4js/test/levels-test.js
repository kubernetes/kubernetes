"use strict";
var vows = require('vows')
, assert = require('assert')
, levels = require('../lib/levels');

function assertThat(level) {
  function assertForEach(assertion, test, otherLevels) {
    otherLevels.forEach(function(other) {
      assertion.call(assert, test.call(level, other));
    });
  }

  return {
    isLessThanOrEqualTo: function(levels) {
      assertForEach(assert.isTrue, level.isLessThanOrEqualTo, levels);
    },
    isNotLessThanOrEqualTo: function(levels) {
      assertForEach(assert.isFalse, level.isLessThanOrEqualTo, levels);
    },
    isGreaterThanOrEqualTo: function(levels) {
      assertForEach(assert.isTrue, level.isGreaterThanOrEqualTo, levels);
    },
    isNotGreaterThanOrEqualTo: function(levels) {
      assertForEach(assert.isFalse, level.isGreaterThanOrEqualTo, levels);
    },
    isEqualTo: function(levels) {
      assertForEach(assert.isTrue, level.isEqualTo, levels);
    },
    isNotEqualTo: function(levels) {
      assertForEach(assert.isFalse, level.isEqualTo, levels);
    }
  };
}

vows.describe('levels').addBatch({
  'values': {
    topic: levels,
    'should define some levels': function(levels) {
      assert.isNotNull(levels.ALL);
      assert.isNotNull(levels.TRACE);
      assert.isNotNull(levels.DEBUG);
      assert.isNotNull(levels.INFO);
      assert.isNotNull(levels.WARN);
      assert.isNotNull(levels.ERROR);
      assert.isNotNull(levels.FATAL);
      assert.isNotNull(levels.MARK);
      assert.isNotNull(levels.OFF);
    },
    'ALL': {
      topic: levels.ALL,
      'should be less than the other levels': function(all) {
        assertThat(all).isLessThanOrEqualTo(
          [ 
            levels.ALL, 
            levels.TRACE, 
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL,
            levels.MARK,
            levels.OFF
          ]
        );
      },
      'should be greater than no levels': function(all) {
        assertThat(all).isNotGreaterThanOrEqualTo(
          [
            levels.TRACE, 
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      },
      'should only be equal to ALL': function(all) {
        assertThat(all).isEqualTo([levels.toLevel("ALL")]);
        assertThat(all).isNotEqualTo(
          [
            levels.TRACE, 
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      }
    },
    'TRACE': {
      topic: levels.TRACE,
      'should be less than DEBUG': function(trace) {
        assertThat(trace).isLessThanOrEqualTo(
          [
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
        assertThat(trace).isNotLessThanOrEqualTo([levels.ALL]);
      },
      'should be greater than ALL': function(trace) {
        assertThat(trace).isGreaterThanOrEqualTo([levels.ALL, levels.TRACE]);
        assertThat(trace).isNotGreaterThanOrEqualTo(
          [
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      },
      'should only be equal to TRACE': function(trace) {
        assertThat(trace).isEqualTo([levels.toLevel("TRACE")]);
        assertThat(trace).isNotEqualTo(
          [
            levels.ALL, 
            levels.DEBUG, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      }
    },
    'DEBUG': {
      topic: levels.DEBUG,
      'should be less than INFO': function(debug) {
        assertThat(debug).isLessThanOrEqualTo(
          [
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
        assertThat(debug).isNotLessThanOrEqualTo([levels.ALL, levels.TRACE]);
      },
      'should be greater than TRACE': function(debug) {
        assertThat(debug).isGreaterThanOrEqualTo([levels.ALL, levels.TRACE]);
        assertThat(debug).isNotGreaterThanOrEqualTo(
          [
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      },
      'should only be equal to DEBUG': function(trace) {
        assertThat(trace).isEqualTo([levels.toLevel("DEBUG")]);
        assertThat(trace).isNotEqualTo(
          [
            levels.ALL, 
            levels.TRACE, 
            levels.INFO, 
            levels.WARN, 
            levels.ERROR, 
            levels.FATAL, 
            levels.MARK,
            levels.OFF
          ]
        );
      }
    },
    'INFO': {
      topic: levels.INFO,
      'should be less than WARN': function(info) {
        assertThat(info).isLessThanOrEqualTo([
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL, 
          levels.MARK,
          levels.OFF
        ]);
        assertThat(info).isNotLessThanOrEqualTo([levels.ALL, levels.TRACE, levels.DEBUG]);
      },
      'should be greater than DEBUG': function(info) {
        assertThat(info).isGreaterThanOrEqualTo([levels.ALL, levels.TRACE, levels.DEBUG]);
        assertThat(info).isNotGreaterThanOrEqualTo([
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL, 
          levels.MARK,
          levels.OFF
        ]);
      },
      'should only be equal to INFO': function(trace) {
        assertThat(trace).isEqualTo([levels.toLevel("INFO")]);
        assertThat(trace).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL, 
          levels.MARK,
          levels.OFF
        ]);
      }
    },
    'WARN': {
      topic: levels.WARN,
      'should be less than ERROR': function(warn) {
        assertThat(warn).isLessThanOrEqualTo([levels.ERROR, levels.FATAL, levels.MARK, levels.OFF]);
        assertThat(warn).isNotLessThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO
        ]);
      },
      'should be greater than INFO': function(warn) {
        assertThat(warn).isGreaterThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO
        ]);
        assertThat(warn).isNotGreaterThanOrEqualTo([levels.ERROR, levels.FATAL, levels.MARK, levels.OFF]);
      },
      'should only be equal to WARN': function(trace) {
        assertThat(trace).isEqualTo([levels.toLevel("WARN")]);
        assertThat(trace).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.ERROR, 
          levels.FATAL, 
          levels.OFF
        ]);
      }
    },
    'ERROR': {
      topic: levels.ERROR,
      'should be less than FATAL': function(error) {
        assertThat(error).isLessThanOrEqualTo([levels.FATAL, levels.MARK, levels.OFF]);
        assertThat(error).isNotLessThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN
        ]);
      },
      'should be greater than WARN': function(error) {
        assertThat(error).isGreaterThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN
        ]);
        assertThat(error).isNotGreaterThanOrEqualTo([levels.FATAL, levels.MARK, levels.OFF]);
      },
      'should only be equal to ERROR': function(trace) {
        assertThat(trace).isEqualTo([levels.toLevel("ERROR")]);
        assertThat(trace).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.FATAL, 
          levels.MARK,
          levels.OFF
        ]);
      }
    },
    'FATAL': {
      topic: levels.FATAL,
      'should be less than OFF': function(fatal) {
        assertThat(fatal).isLessThanOrEqualTo([levels.MARK, levels.OFF]);
        assertThat(fatal).isNotLessThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR
        ]);
      },
      'should be greater than ERROR': function(fatal) {
        assertThat(fatal).isGreaterThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR
       ]);
        assertThat(fatal).isNotGreaterThanOrEqualTo([levels.MARK, levels.OFF]);
      },
      'should only be equal to FATAL': function(fatal) {
        assertThat(fatal).isEqualTo([levels.toLevel("FATAL")]);
        assertThat(fatal).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR,
          levels.MARK, 
          levels.OFF
        ]);
      }
    },
    'MARK': {
      topic: levels.MARK,
      'should be less than OFF': function(mark) {
        assertThat(mark).isLessThanOrEqualTo([levels.OFF]);
        assertThat(mark).isNotLessThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.FATAL, 
          levels.ERROR
        ]);
      },
      'should be greater than FATAL': function(mark) {
        assertThat(mark).isGreaterThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR,
          levels.FATAL
       ]);
        assertThat(mark).isNotGreaterThanOrEqualTo([levels.OFF]);
      },
      'should only be equal to MARK': function(mark) {
        assertThat(mark).isEqualTo([levels.toLevel("MARK")]);
        assertThat(mark).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR,
          levels.FATAL, 
          levels.OFF
        ]);
      }
    },
    'OFF': {
      topic: levels.OFF,
      'should not be less than anything': function(off) {
        assertThat(off).isNotLessThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL,
          levels.MARK
        ]);
      },
      'should be greater than everything': function(off) {
        assertThat(off).isGreaterThanOrEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL,
          levels.MARK
        ]);
      },
      'should only be equal to OFF': function(off) {
        assertThat(off).isEqualTo([levels.toLevel("OFF")]);
        assertThat(off).isNotEqualTo([
          levels.ALL, 
          levels.TRACE, 
          levels.DEBUG, 
          levels.INFO, 
          levels.WARN, 
          levels.ERROR, 
          levels.FATAL,
          levels.MARK
        ]);
      }
    }
  },
  'isGreaterThanOrEqualTo': {
    topic: levels.INFO,
    'should handle string arguments': function(info) {
      assertThat(info).isGreaterThanOrEqualTo(["all", "trace", "debug"]);
      assertThat(info).isNotGreaterThanOrEqualTo(['warn', 'ERROR', 'Fatal', 'MARK', 'off']);
    }
  },
  'isLessThanOrEqualTo': {
    topic: levels.INFO,
    'should handle string arguments': function(info) {
      assertThat(info).isNotLessThanOrEqualTo(["all", "trace", "debug"]);
      assertThat(info).isLessThanOrEqualTo(['warn', 'ERROR', 'Fatal', 'MARK', 'off']);
    }
  },
  'isEqualTo': {
    topic: levels.INFO,
    'should handle string arguments': function(info) {
      assertThat(info).isEqualTo(["info", "INFO", "iNfO"]);
    }
  },
  'toLevel': {
    'with lowercase argument': {
      topic: levels.toLevel("debug"),
      'should take the string and return the corresponding level': function(level) {
        assert.equal(level, levels.DEBUG);
      }
    },
    'with uppercase argument': {
      topic: levels.toLevel("DEBUG"),
      'should take the string and return the corresponding level': function(level) {
        assert.equal(level, levels.DEBUG);
      }
    },
    'with varying case': {
      topic: levels.toLevel("DeBuG"),
      'should take the string and return the corresponding level': function(level) {
        assert.equal(level, levels.DEBUG);
      }
    },
    'with unrecognised argument': {
      topic: levels.toLevel("cheese"),
      'should return undefined': function(level) {
        assert.isUndefined(level);
      }
    },
    'with unrecognised argument and default value': {
      topic: levels.toLevel("cheese", levels.DEBUG),
      'should return default value': function(level) {
        assert.equal(level, levels.DEBUG);
      }
    }
  }
}).export(module);
