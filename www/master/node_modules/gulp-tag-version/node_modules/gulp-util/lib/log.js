var colors = require('./colors');
var date = require('./date');

module.exports = function(){
  var time = '['+colors.grey(date(new Date(), 'HH:MM:ss'))+']';
  var args = Array.prototype.slice.call(arguments);
  args.unshift(time);
  console.log.apply(console, args);
  return this;
};
