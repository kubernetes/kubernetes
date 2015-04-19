"use strict";
/* jshint loopfunc: true */
// This test shows an asymmetry between setLevel and isLevelEnabled 
// (in log4js-node@0.4.3 and earlier):
// 1) setLevel("foo") works, but setLevel(log4js.levels.foo) silently 
//    does not (sets the level to TRACE).
// 2) isLevelEnabled("foo") works as does isLevelEnabled(log4js.levels.foo).
//

// Basic set up
var vows = require('vows');
var assert = require('assert');
var log4js = require('../lib/log4js');
var logger = log4js.getLogger('test-setLevel-asymmetry');

// uncomment one or other of the following to see progress (or not) while running the tests
// var showProgress = console.log;
var showProgress = function() {};


// Define the array of levels as string to iterate over.
var strLevels= ['Trace','Debug','Info','Warn','Error','Fatal'];

var log4jsLevels =[];
// populate an array with the log4js.levels that match the strLevels.
// Would be nice if we could iterate over log4js.levels instead, 
// but log4js.levels.toLevel prevents that for now.
strLevels.forEach(function(l) {
  log4jsLevels.push(log4js.levels.toLevel(l));
});


// We are going to iterate over this object's properties to define an exhaustive list of vows.
var levelTypes = {
  'string': strLevels,
  'log4js.levels.level': log4jsLevels,
};

// Set up the basic vows batch for this test
var batch = {
  setLevel: {
  }
};

showProgress('Populating batch object...');

// Populating the batch object programmatically,
// as I don't have the patience to manually populate it with 
// the (strLevels.length x levelTypes.length) ^ 2 = 144 possible test combinations
for (var type in levelTypes) {
  var context = 'is called with a '+type;
  var levelsToTest = levelTypes[type];
  showProgress('Setting up the vows context for '+context);

  batch.setLevel[context]= {};
  levelsToTest.forEach( function(level) {
    var subContext = 'of '+level;
    var log4jsLevel=log4js.levels.toLevel(level.toString());

    showProgress('Setting up the vows sub-context for '+subContext);
    batch.setLevel[context][subContext] = {topic: level};
    for (var comparisonType in levelTypes) {
      levelTypes[comparisonType].forEach(function(comparisonLevel) {
        var t = type;
        var ct = comparisonType;
        var expectedResult = log4jsLevel.isLessThanOrEqualTo(comparisonLevel);
        var vow = 'isLevelEnabled(' + comparisonLevel + 
          ') called with a ' + comparisonType + 
          ' should return ' + expectedResult;
        showProgress('Setting up the vows vow for '+vow);

        batch.setLevel[context][subContext][vow] = function(levelToSet) {
          logger.setLevel(levelToSet);
          showProgress(
            '*** Checking setLevel( ' + level + 
              ' ) of type ' + t + 
              ', and isLevelEnabled( ' + comparisonLevel + 
              ' ) of type ' + ct + '. Expecting: ' + expectedResult
          );
          assert.equal(
            logger.isLevelEnabled(comparisonLevel), 
            expectedResult, 
            'Failed: calling setLevel( ' + level + 
              ' ) with type ' + type + 
              ', isLevelEnabled( ' + comparisonLevel + 
              ' ) of type ' + comparisonType + 
              ' did not return ' + expectedResult
          );
        };
      });
    }
  });

}

showProgress('Running tests...');

vows.describe('log4js setLevel asymmetry fix').addBatch(batch).export(module);


