var ent = require('ent');
console.log(ent.encode('<span>©moo</span>'))
console.log(ent.decode('&pi; &amp; &rho;'));
