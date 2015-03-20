/* jshint -W079 */
/* Related to https://github.com/linnovate/mean/issues/898 */
'use strict';

/**
 * Module dependencies.
 */
var expect = require('expect.js'),
  mongoose = require('mongoose'),
  User = mongoose.model('User'),
  Article = mongoose.model('Article');

/**
 * Globals
 */
var user;
var article;

/**
 * Test Suites
 */
describe('<Unit Test>', function() {
  describe('Model Article:', function() {
    beforeEach(function(done) {
      this.timeout(10000);
      user = new User({
        name: 'Full name',
        email: 'test@test.com',
        username: 'user',
        password: 'password'
      });
      user.save(function() {
        article = new Article({
          title: 'Article Title',
          content: 'Article Content',
          user: user
        });
        done();
      });


    });
    describe('Method Save', function() {

      it('should be able to save without problems', function(done) {
        this.timeout(10000);

        return article.save(function(err, data) {
          expect(err).to.be(null);
          expect(data.title).to.equal('Article Title');
          expect(data.content).to.equal('Article Content');
          expect(data.user.length).to.not.equal(0);
          expect(data.created.length).to.not.equal(0);
          done();
        });

      });

      it('should be able to show an error when try to save without title', function(done) {
        this.timeout(10000);
        article.title = '';

        return article.save(function(err) {
          expect(err).to.not.be(undefined);
          done();
        });
      });

      it('should be able to show an error when try to save without content', function(done) {
        this.timeout(10000);
        article.content = '';

        return article.save(function(err) {
          expect(err).to.not.be(undefined);
          done();
        });
      });

      it('should be able to show an error when try to save without user', function(done) {
        this.timeout(10000);
        article.user = {};

        return article.save(function(err) {
          expect(err).to.not.be(undefined);
          done();
        });
      });

    });

    afterEach(function(done) {
      this.timeout(10000);
      article.remove(function() {
        user.remove(done);
      });
    });
  });
});
