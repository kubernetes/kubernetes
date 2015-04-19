"use strict";
var vows = require('vows')
, assert = require('assert')
, dateFormat = require('../lib/date_format');

function createFixedDate() {
  return new Date(2010, 0, 11, 14, 31, 30, 5);
}

vows.describe('date_format').addBatch({
  'Date extensions': {
    topic: createFixedDate,
    'should format a date as string using a pattern': function(date) {
      assert.equal(
        dateFormat.asString(dateFormat.DATETIME_FORMAT, date),
        "11 01 2010 14:31:30.005"
      );
    },
    'should default to the ISO8601 format': function(date) {
      assert.equal(
        dateFormat.asString(date),
        '2010-01-11 14:31:30.005'
      );
    },
    'should provide a ISO8601 with timezone offset format': function() {
      var date = createFixedDate();
      date.setMinutes(date.getMinutes() - date.getTimezoneOffset() - 660);
      date.getTimezoneOffset = function() { return -660; };
      assert.equal(
        dateFormat.asString(dateFormat.ISO8601_WITH_TZ_OFFSET_FORMAT, date),
        "2010-01-11T14:31:30+1100"
      );
      date = createFixedDate();
      date.setMinutes(date.getMinutes() - date.getTimezoneOffset() + 120);
      date.getTimezoneOffset = function() { return 120; };
      assert.equal(
        dateFormat.asString(dateFormat.ISO8601_WITH_TZ_OFFSET_FORMAT, date),
        "2010-01-11T14:31:30-0200"
      );

    },
    'should provide a just-the-time format': function(date) {
      assert.equal(
        dateFormat.asString(dateFormat.ABSOLUTETIME_FORMAT, date),
        '14:31:30.005'
      );
    },
    'should provide a custom format': function() {
      var date = createFixedDate();
      date.setMinutes(date.getMinutes() - date.getTimezoneOffset() + 120);
      date.getTimezoneOffset = function() { return 120; };
      assert.equal(
        dateFormat.asString("O.SSS.ss.mm.hh.dd.MM.yy", date),
        '-0200.005.30.31.14.11.01.10'
      );
    }
  }
}).export(module);
