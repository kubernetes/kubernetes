'use strict';

// User routes use users controller
var users = require('../controllers/users'),
    config = require('meanio').loadConfig();

module.exports = function(MeanUser, app, auth, database, passport) {

  app.route('/logout')
    .get(users.signout);
  app.route('/users/me')
    .get(users.me);

  // Setting up the users api
  app.route('/register')
    .post(users.create);

  app.route('/forgot-password')
    .post(users.forgotpassword);

  app.route('/reset/:token')
    .post(users.resetpassword);

  // Setting up the userId param
  app.param('userId', users.user);

  // AngularJS route to check for authentication
  app.route('/loggedin')
    .get(function(req, res) {
      res.send(req.isAuthenticated() ? req.user : '0');
    });

  // Setting the local strategy route
  app.route('/login')
    .post(passport.authenticate('local', {
      failureFlash: true
    }), function(req, res) {
      res.send({
        user: req.user,
        redirect: (req.user.roles.indexOf('admin') !== -1) ? req.get('referer') : false
      });
    });

  // AngularJS route to get config of social buttons
  app.route('/get-config')
    .get(function (req, res) {
      // To avoid displaying unneccesary social logins
      var clientIdProperty = 'clientID';
      var defaultPrefix = 'DEFAULT_';
      var socialNetworks = ['facebook','linkedin','twitter','github','google']; //ugly hardcoding :(
      var configuredApps = {};
      for (var network in socialNetworks){
        var netObject = config[socialNetworks[network]];
        if ( netObject.hasOwnProperty(clientIdProperty) ) {
              if (netObject[clientIdProperty].indexOf(defaultPrefix) === -1 ){
                configuredApps[socialNetworks[network]] = true ;
              }
        }
      }
      res.send(configuredApps);
    });

  // Setting the facebook oauth routes
  app.route('/auth/facebook')
    .get(passport.authenticate('facebook', {
      scope: ['email', 'user_about_me'],
      failureRedirect: '#!/login'
    }), users.signin);

  app.route('/auth/facebook/callback')
    .get(passport.authenticate('facebook', {
      failureRedirect: '#!/login'
    }), users.authCallback);

  // Setting the github oauth routes
  app.route('/auth/github')
    .get(passport.authenticate('github', {
      failureRedirect: '#!/login'
    }), users.signin);

  app.route('/auth/github/callback')
    .get(passport.authenticate('github', {
      failureRedirect: '#!/login'
    }), users.authCallback);

  // Setting the twitter oauth routes
  app.route('/auth/twitter')
    .get(passport.authenticate('twitter', {
      failureRedirect: '#!/login'
    }), users.signin);

  app.route('/auth/twitter/callback')
    .get(passport.authenticate('twitter', {
      failureRedirect: '#!/login'
    }), users.authCallback);

  // Setting the google oauth routes
  app.route('/auth/google')
    .get(passport.authenticate('google', {
      failureRedirect: '#!/login',
      scope: [
        'https://www.googleapis.com/auth/userinfo.profile',
        'https://www.googleapis.com/auth/userinfo.email'
      ]
    }), users.signin);

  app.route('/auth/google/callback')
    .get(passport.authenticate('google', {
      failureRedirect: '#!/login'
    }), users.authCallback);

  // Setting the linkedin oauth routes
  app.route('/auth/linkedin')
    .get(passport.authenticate('linkedin', {
      failureRedirect: '#!/login',
      scope: ['r_emailaddress']
    }), users.signin);

  app.route('/auth/linkedin/callback')
    .get(passport.authenticate('linkedin', {
      failureRedirect: '#!/login'
    }), users.authCallback);

};
