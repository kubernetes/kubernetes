/**
 * Base interface class other can inherits from
 */

var _ = require("lodash");
var rx = require("rx");
var util = require("util");
var utils = require("../utils/utils");
var Base = require("./baseUI");

var inquirer = require("../inquirer");


/**
 * Module exports
 */

module.exports = PromptUI;


/**
 * Constructor
 */

function PromptUI( prompts ) {
  Base.call(this);
  this.prompts = prompts;
}
util.inherits( PromptUI, Base );

PromptUI.prototype.run = function( questions, allDone ) {
  // Keep global reference to the answers
  this.answers = {};
  this.completed = allDone;

  // Make sure questions is an array.
  if ( _.isPlainObject(questions) ) {
    questions = [questions];
  }

  // Create an observable, unless we received one as parameter.
  // Note: As this is a public interface, we cannot do an instanceof check as we won't
  // be using the exact same object in memory.
  var obs = _.isArray( questions ) ? rx.Observable.fromArray( questions ) : questions;

  // Start running the questions
  this.process = obs.concatMap( this.processQuestion.bind(this) );

  this.process.forEach(
    function() {},
    function( err ) {
      throw err;
    },
    this.onCompletion.bind(this)
  );

  return this.process;
};


/**
 * Once all prompt are over
 */

PromptUI.prototype.onCompletion = function() {
  this.close();

  if ( _.isFunction(this.completed) ) {
    this.completed( this.answers );
  }
};

PromptUI.prototype.processQuestion = function( question ) {
  return rx.Observable.defer(function() {
    var obs = rx.Observable.create(function(obs) {
      obs.onNext( question );
      obs.onCompleted();
    });

    return obs
      .concatMap( this.setDefaultType.bind(this) )
      .concatMap( this.filterIfRunnable.bind(this) )
      .concatMap( utils.fetchAsyncQuestionProperty.bind( null, question, "message", this.answers ) )
      .concatMap( utils.fetchAsyncQuestionProperty.bind( null, question, "default", this.answers ) )
      .concatMap( utils.fetchAsyncQuestionProperty.bind( null, question, "choices", this.answers ) )
      .concatMap( this.fetchAnswer.bind(this) );
  }.bind(this));
};

PromptUI.prototype.fetchAnswer = function( question ) {
  var Prompt = this.prompts[question.type];
  var prompt = new Prompt( question, this.rl, this.answers );
  var answers = this.answers;
  return utils.createObservableFromAsync(function() {
    var done = this.async();
    prompt.run(function( answer ) {
      answers[question.name] = answer;
      done({ name: question.name, answer: answer });
    });
  });
};

PromptUI.prototype.setDefaultType = function( question ) {
  // Default type to input
  if ( !this.prompts[question.type] ) {
    question.type = "input";
  }
  return rx.Observable.defer(function() {
    return rx.Observable.return( question );
  });
};

PromptUI.prototype.filterIfRunnable = function( question ) {
  if ( !_.isFunction(question.when) ) return rx.Observable.return(question);

  var answers = this.answers;
  return rx.Observable.defer(function() {
    return rx.Observable.create(function( obs ) {
      utils.runAsync( question.when, function( shouldRun ) {
        if ( shouldRun ) {
          obs.onNext( question );
        }
        obs.onCompleted();
      }, answers );
    });
  });
};
