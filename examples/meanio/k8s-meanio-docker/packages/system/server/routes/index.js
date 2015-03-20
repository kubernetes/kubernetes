'use strict';

var mean = require('meanio');

module.exports = function(System, app, auth, database) {

  // Home route
  var index = require('../controllers/index');
  app.route('/')
    .get(index.render);


  app.get('/*',function(req,res,next){
        res.header('workerID' , JSON.stringify(mean.options.workerid) );
        next(); // http://expressjs.com/guide.html#passing-route control
  });
};
