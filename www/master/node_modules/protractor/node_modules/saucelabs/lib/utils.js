var slice = Array.prototype.slice;

exports.extend = function (obj) {
  slice.call(arguments, 1).forEach(function (props) {
    var prop;
    for (prop in props) {
      obj[prop] = props[prop];
    }
  });
  return obj;
};

exports.replace = function (str, values) {
  var name, value;
  for (name in values) {
    value = values[name];
    str = str.replace(new RegExp(':' + name, 'g'), value);
  }
  return str;
};
