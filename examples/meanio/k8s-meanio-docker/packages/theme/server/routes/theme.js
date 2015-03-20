'use strict';

// The Package is passed automatically as first parameter
module.exports = function(Theme, app, auth, database) {

  app.get('/theme/example/anyone', function(req, res, next) {
    res.send('Anyone can access this');
  });

  app.get('/theme/example/auth', auth.requiresLogin, function(req, res, next) {
    res.send('Only authenticated users can access this');
  });

  app.get('/theme/example/admin', auth.requiresAdmin, function(req, res, next) {
    res.send('Only users with Admin role can access this');
  });

  app.get('/theme/example/render', function(req, res, next) {
    Theme.render('index', {
      package: 'theme'
    }, function(err, html) {
      //Rendering a view from the Package server/views
      res.send(html);
    });
  });
};
