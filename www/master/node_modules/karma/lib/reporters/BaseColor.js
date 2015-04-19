require('colors');

var BaseColorReporter = function() {
  this.USE_COLORS = true;

  this.LOG_SINGLE_BROWSER = '%s: ' + '%s'.cyan + '\n';
  this.LOG_MULTI_BROWSER = '%s %s: ' + '%s'.cyan + '\n';

  this.SPEC_FAILURE = '%s %s FAILED'.red + '\n';
  this.SPEC_SLOW = '%s SLOW %s: %s'.yellow + '\n';
  this.ERROR = '%s ERROR'.red + '\n';

  this.FINISHED_ERROR = ' ERROR'.red;
  this.FINISHED_SUCCESS = ' SUCCESS'.green;
  this.FINISHED_DISCONNECTED = ' DISCONNECTED'.red;

  this.X_FAILED = ' (%d FAILED)'.red;

  this.TOTAL_SUCCESS = 'TOTAL: %d SUCCESS'.green + '\n';
  this.TOTAL_FAILED = 'TOTAL: %d FAILED, %d SUCCESS'.red + '\n';
};


// PUBLISH
module.exports = BaseColorReporter;
