'use strict';

var articles = require('../controllers/articles');

// Article authorization helpers
var hasAuthorization = function(req, res, next) {
  if (!req.user.isAdmin && req.article.user.id !== req.user.id) {
    return res.status(401).send('User is not authorized');
  }
  next();
};

module.exports = function(Articles, app, auth) {

  app.route('/articles')
    .get(articles.all)
    .post(auth.requiresLogin, articles.create);
  app.route('/articles/:articleId')
    .get(auth.isMongoId, articles.show)
    .put(auth.isMongoId, auth.requiresLogin, hasAuthorization, articles.update)
    .delete(auth.isMongoId, auth.requiresLogin, hasAuthorization, articles.destroy);

  // Finish with setting up the articleId param
  app.param('articleId', articles.article);
};
