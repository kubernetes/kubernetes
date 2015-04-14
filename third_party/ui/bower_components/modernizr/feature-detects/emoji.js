// Requires a Modernizr build with `canvastext` included
// http://www.modernizr.com/download/#-canvas-canvastext
Modernizr.addTest('emoji', function() {
  if (!Modernizr.canvastext) return false;
  var node = document.createElement('canvas'),
      ctx = node.getContext('2d');
  ctx.textBaseline = 'top';
  ctx.font = '32px Arial';
  ctx.fillText('\ud83d\ude03', 0, 0); // "smiling face with open mouth" emoji
  return ctx.getImageData(16, 16, 1, 1).data[0] !== 0;
});