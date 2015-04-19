function RGB(red, green, blue) {
  this.red = red;
  this.green = green;
  this.blue = blue;
}

RGB.prototype.toHex = function () {
  var red = Math.max(0, Math.min(~~this.red, 255));
  var green = Math.max(0, Math.min(~~this.green, 255));
  var blue = Math.max(0, Math.min(~~this.blue, 255));

  // Credit: Asen  http://jsbin.com/UPUmaGOc/2/edit?js,console
  return '#' + ('00000' + (red << 16 | green << 8 | blue).toString(16)).slice(-6);
};

module.exports = RGB;
