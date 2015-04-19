var chalk = require('chalk');
var dateformat = require('dateformat');

module.exports = function(){
  var time = '['+chalk.grey(dateformat(new Date(), 'HH:MM:ss'))+']';
  var args = Array.prototype.slice.call(arguments);
  args.unshift(time);
  console.log.apply(console, args);
  return this;
};
