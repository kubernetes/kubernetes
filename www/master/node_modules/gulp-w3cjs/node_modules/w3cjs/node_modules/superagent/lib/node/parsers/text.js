
module.exports = function(res, fn){
  res.text = '';
  res.setEncoding('utf8');
  res.on('data', function(chunk){ res.text += chunk; });
  res.on('end', fn);
};