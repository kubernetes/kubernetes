class Expression {
  nodeType: ExpressionType;

  constructor(nodeType: ExpressionType) {
    this.nodeType = nodeType;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    throw new Error("not implemented");
  }

  static Constant(value: any): ConstantExpression {
    return new ConstantExpression(value);
  }

  static Parameter(name: string): ParameterExpression {
    return new ParameterExpression(name);
  }

  static Condition(test: Expression, ifTrue: Expression, ifFalse: Expression): ConditionalExpression {
    return new ConditionalExpression(test, ifTrue, ifFalse);
  }

  static Add(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Add, left, right);
  }

  static Subtract(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Subtract, left, right);
  }

  static Multiply(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Multiply, left, right);
  }

  static Divide(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Divide, left, right);
  }

  static Modulo(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Modulo, left, right);
  }

  static And(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.And, left, right);
  }

  static AndAlso(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.AndAlso, left, right);
  }

  static Or(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Or, left, right);
  }

  static OrElse(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.OrElse, left, right);
  }

  static ExclusiveOr(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.ExclusiveOr, left, right);
  }

  static Equal(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.Equal, left, right);
  }

  static NotEqual(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.NotEqual, left, right);
  }

  static LessThan(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.LessThan, left, right);
  }

  static LessThanOrEqual(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.LessThanOrEqual, left, right);
  }

  static GreaterThan(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.GreaterThan, left, right);
  }

  static GreaterThanOrEqual(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.GreaterThanOrEqual, left, right);
  }

  static LeftShift(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.LeftShift, left, right);
  }

  static RightShift(left: Expression, right: Expression): BinaryExpression {
    return new BinaryExpression(ExpressionType.RightShift, left, right);
  }

  static Not(operand: Expression): UnaryExpression {
    return new UnaryExpression(ExpressionType.Not, operand);
  }

  static UnaryPlus(operand: Expression): UnaryExpression {
    return new UnaryExpression(ExpressionType.UnaryPlus, operand);
  }

  static Negate(operand: Expression): UnaryExpression {
    return new UnaryExpression(ExpressionType.Negate, operand);
  }

  static OnesComplement(operand: Expression): UnaryExpression {
    return new UnaryExpression(ExpressionType.OnesComplement, operand);
  }

  static Lambda<T extends Function>(body: Expression, ...parameters: ParameterExpression[]): LambdaExpression<T> {
    return new LambdaExpression<T>(body, parameters);
  }

  static Invoke(expression: Expression, ...args: Expression[]): InvocationExpression {
    return new InvocationExpression(expression, args);
  }

  static New(typeName: string, ...args: Expression[]): NewExpression {
    return new NewExpression(typeName, args);
  }

  static Call(obj: Expression, methodName: string, ...args: Expression[]): CallExpression {
    return new CallExpression(obj, methodName, args);
  }

  static Member(obj: Expression, memberName: string): MemberExpression {
    return new MemberExpression(obj, memberName);
  }

  static Index(obj: Expression, ...args: Expression[]): IndexExpression {
    return new IndexExpression(obj, args);
  }
}

class ExpressionVisitor {
  Visit(node: Expression): Expression {
    if (node === null) {
      return null;
    }
    return node.Accept(this);
  }

  VisitConstant(node: ConstantExpression): Expression {
    return node;
  }

  VisitParameter(node: ParameterExpression): Expression {
    return node;
  }

  VisitBinary(node: BinaryExpression): Expression {
    return node.Update(this.Visit(node.left), this.Visit(node.right));
  }

  VisitUnary(node: UnaryExpression): Expression {
    return node.Update(this.Visit(node.operand));
  }

  VisitConditional(node: ConditionalExpression): Expression {
    return node.Update(this.Visit(node.test), this.Visit(node.ifTrue), this.Visit(node.ifFalse));
  }

  VisitLambda<T extends Function>(node: LambdaExpression<T>): Expression {
    return node.Update(this.Visit(node.body), this.VisitMany(node.parameters));
  }

  VisitInvoke(node: InvocationExpression): Expression {
    return node.Update(this.Visit(node.expression), this.VisitMany(node.args));
  }

  VisitCall(node: CallExpression): Expression {
    return node.Update(this.Visit(node.obj), this.VisitMany(node.args));
  }

  VisitNew(node: NewExpression): Expression {
    return node.Update(this.VisitMany(node.args));
  }

  VisitMember(node: MemberExpression): Expression {
    return node.Update(this.Visit(node.obj));
  }

  VisitIndex(node: IndexExpression): Expression {
    return node.Update(this.Visit(node.obj), this.VisitMany(node.args));
  }

  VisitMany<T extends Expression>(nodes: T[]): T[] {
    var res = new Array<T>(nodes.length);

    for (var i = 0; i < nodes.length; i++) {
      var oldNode = nodes[i];
      var newNode = <T>this.Visit(oldNode);
      res[i] = newNode;
    }

    return res;
  }
}

class ConstantExpression extends Expression {
  _value: any;

  constructor(value: any) {
    super(ExpressionType.Constant);
    this._value = value;
  }

  get value(): any {
    return this._value;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitConstant(this);
  }
}

class ParameterExpression extends Expression {
  _name: string;

  constructor(name: string) {
    super(ExpressionType.Parameter);
    this._name = name;
  }

  get name(): string {
    return this._name;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitParameter(this);
  }
}

class UnaryExpression extends Expression {
  _operand: Expression;

  constructor(nodeType: ExpressionType, operand: Expression) {
    super(nodeType);
    this._operand = operand;
  }

  get operand(): Expression {
    return this._operand;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitUnary(this);
  }

  Update(operand: Expression): UnaryExpression {
    if (operand !== this._operand) {
      return new UnaryExpression(this.nodeType, operand);
    }

    return this;
  }
}

class BinaryExpression extends Expression {
  _left: Expression;
  _right: Expression;

  constructor(nodeType: ExpressionType, left: Expression, right: Expression) {
    super(nodeType);
    this._left = left;
    this._right = right;
  }

  get left(): Expression {
    return this._left;
  }

  get right(): Expression {
    return this._right;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitBinary(this);
  }

  Update(left: Expression, right: Expression): BinaryExpression {
    if (left !== this._left || right !== this._right) {
      return new BinaryExpression(this.nodeType, left, right);
    }

    return this;
  }
}

class ConditionalExpression extends Expression {
  _test: Expression;
  _ifTrue: Expression;
  _ifFalse: Expression;

  constructor(test: Expression, ifTrue: Expression, ifFalse: Expression) {
    super(ExpressionType.Condition);
    this._test = test;
    this._ifTrue = ifTrue;
    this._ifFalse = ifFalse;
  }

  get test(): Expression {
    return this._test;
  }

  get ifTrue(): Expression {
    return this._ifTrue;
  }

  get ifFalse(): Expression {
    return this._ifTrue;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitConditional(this);
  }

  Update(test: Expression, ifTrue: Expression, ifFalse: Expression): ConditionalExpression {
    if (test !== this._test || ifTrue !== this._ifTrue || ifFalse !== this._ifFalse) {
      return new ConditionalExpression(test, ifTrue, ifFalse);
    }

    return this;
  }
}

class LambdaExpression<T extends Function> extends Expression {
  _body: Expression;
  _parameters: ParameterExpression[];

  constructor(body: Expression, parameters: ParameterExpression[]) {
    super(ExpressionType.Lambda);
    this._body = body;
    this._parameters = parameters;
  }

  get body(): Expression {
    return this._body;
  }

  get parameters(): ParameterExpression[] {
    return this._parameters;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitLambda<T>(this);
  }

  Update(body: Expression, parameters: ParameterExpression[]): LambdaExpression<T> {
    if (body !== this._body || parameters !== this._parameters) {
      return new LambdaExpression<T>(body, parameters);
    }

    return this;
  }

  Compile(): T {
    var comp = new LambdaCompiler();
    comp.Visit(this);
    var code = comp.code;
    code = code.replace(/\"/g, "\\\""); // TODO: more escape sequences
    code = "new Function(\"return " + code + ";\")";
    code = code.replace(/\r?\n|\r/g, "");
    alert(code);
    return <T>eval(code)();
  }
}

class InvocationExpression extends Expression {
  _expression: Expression;
  _args: Expression[];

  constructor(expression: Expression, args: Expression[]) {
    super(ExpressionType.Invoke);
    this._expression = expression;
    this._args = args;
  }

  get expression(): Expression {
    return this._expression;
  }

  get args(): Expression[] {
    return this._args;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitInvoke(this);
  }

  Update(expression: Expression, args: Expression[]): InvocationExpression {
    if (expression !== this._expression || args !== this._args) {
      return new InvocationExpression(expression, args);
    }

    return this;
  }
}

class CallExpression extends Expression {
  _expression: Expression;
  _method: string;
  _args: Expression[];

  constructor(expression: Expression, methodName: string, args: Expression[]) {
    super(ExpressionType.Call);
    this._expression = expression;
    this._method = methodName;
    this._args = args;
  }

  get obj(): Expression {
    return this._expression;
  }

  get method(): string {
    return this._method;
  }

  get args(): Expression[] {
    return this._args;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitCall(this);
  }

  Update(expression: Expression, args: Expression[]): CallExpression {
    if (expression !== this._expression || args !== this._args) {
      return new CallExpression(expression, this._method, args);
    }

    return this;
  }
}

class IndexExpression extends Expression {
  _expression: Expression;
  _args: Expression[];

  constructor(expression: Expression, args: Expression[]) {
    super(ExpressionType.Index);
    this._expression = expression;
    this._args = args;
  }

  get obj(): Expression {
    return this._expression;
  }

  get args(): Expression[] {
    return this._args;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitIndex(this);
  }

  Update(expression: Expression, args: Expression[]): IndexExpression {
    if (expression !== this._expression || args !== this._args) {
      return new IndexExpression(expression, args);
    }

    return this;
  }
}

class NewExpression extends Expression {
  _type: string;
  _args: Expression[];

  constructor(typeName: string, args: Expression[]) {
    super(ExpressionType.New);
    this._type = typeName;
    this._args = args;
  }

  get type(): string {
    return this._type;
  }

  get args(): Expression[] {
    return this._args;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitNew(this);
  }

  Update(args: Expression[]): NewExpression {
    if (args !== this._args) {
      return new NewExpression(this._type, args);
    }

    return this;
  }
}

class MemberExpression extends Expression {
  _obj: Expression;
  _member: string;

  constructor(obj: Expression, memberName: string) {
    super(ExpressionType.Member);
    this._obj = obj;
    this._member = memberName;
  }

  get obj(): Expression {
    return this._obj;
  }

  get member(): string {
    return this._member;
  }

  Accept(visitor: ExpressionVisitor): Expression {
    return visitor.VisitMember(this);
  }

  Update(obj: Expression): MemberExpression {
    if (obj !== this._obj) {
      return new MemberExpression(obj, this._member);
    }

    return this;
  }
}

class LambdaCompiler extends ExpressionVisitor {
  _stack: string[];

  constructor() {
    super();
    this._stack = new Array<string>();
  }

  get code(): string {
    if (this._stack.length != 1)
      throw new Error("invalid code generation");

    return this._stack[0];
  }

  VisitConstant(node: ConstantExpression): Expression {
    var value = "";

    if (typeof node.value == "string") {
      value = "\"" + node.value + "\""; // TODO: escape characters
    }
    else if (node.value instanceof Array) {
      value = JSON.stringify(node.value);
    }
    else if (node.value === undefined) {
      value = "undefined";
    }
    else {
      value = node.value.toString(); // TODO
    }

    this._stack.push(value);

    return node;
  }

  VisitUnary(node: UnaryExpression): Expression {
    this.Visit(node.operand);

    var o = this._stack.pop();
    var i = "";

    switch (node.nodeType) {
      case ExpressionType.Negate:
        i = "-";
        break;
      case ExpressionType.UnaryPlus:
        i = "+";
        break;
      case ExpressionType.Not:
        i = "!";
        break;
      case ExpressionType.OnesComplement:
        i = "~";
        break;
    }

    var res = "(" + i + "" + o + ")";
    this._stack.push(res);

    return node;
  }

  VisitBinary(node: BinaryExpression): Expression {
    this.Visit(node.left);
    this.Visit(node.right);

    var r = this._stack.pop();
    var l = this._stack.pop();
    var i = "";

    switch (node.nodeType) {
      case ExpressionType.Add:
        i = "+";
        break;
      case ExpressionType.Subtract:
        i = "-";
        break;
      case ExpressionType.Multiply:
        i = "*";
        break;
      case ExpressionType.Divide:
        i = "/";
        break;
      case ExpressionType.Modulo:
        i = "%";
        break;
      case ExpressionType.And:
        i = "&";
        break;
      case ExpressionType.Or:
        i = "|";
        break;
      case ExpressionType.AndAlso:
        i = "&&";
        break;
      case ExpressionType.OrElse:
        i = "||";
        break;
      case ExpressionType.ExclusiveOr:
        i = "^";
        break;
      case ExpressionType.Equal:
        i = "===";
        break;
      case ExpressionType.NotEqual:
        i = "!==";
        break;
      case ExpressionType.LessThan:
        i = "<";
        break;
      case ExpressionType.LessThanOrEqual:
        i = "<=";
        break;
      case ExpressionType.GreaterThan:
        i = ">";
        break;
      case ExpressionType.GreaterThanOrEqual:
        i = ">=";
        break;
      case ExpressionType.LeftShift:
        i = "<<";
        break;
      case ExpressionType.RightShift:
        i = ">>";
        break;
    }

    var res = "(" + l + " " + i + " " + r + ")";
    this._stack.push(res);

    return node;
  }

  VisitConditional(node: ConditionalExpression): Expression {
    this.Visit(node.test);
    this.Visit(node.ifTrue);
    this.Visit(node.ifFalse);

    var f = this._stack.pop();
    var t = this._stack.pop();
    var c = this._stack.pop();

    var res = "(" + c + " ? " + t + " : " + f + ")";

    this._stack.push(res);

    return node;
  }

  VisitParameter(node: ParameterExpression): Expression {
    this._stack.push(node.name);

    return node;
  }

  VisitLambda<T extends Function>(node: LambdaExpression<T>): Expression {
    this.VisitMany(node.parameters);
    this.Visit(node.body);

    var body = this._stack.pop();

    var n = node.parameters.length;
    var args = new Array<string>(n);
    for (var i = 0; i < n; i++) {
      args[n - i - 1] = this._stack.pop();
    }

    var allArgs = args.join(", ");

    var res = "function(" + allArgs + ") { return " + body + "; }";

    this._stack.push(res);

    return node;
  }

  VisitInvoke(node: InvocationExpression): Expression {
    this.Visit(node.expression);
    this.VisitMany(node.args);

    var n = node.args.length;
    var args = new Array<string>(n);
    for (var i = 0; i < n; i++) {
      args[n - i - 1] = this._stack.pop();
    }

    var argList = args.join(", ");

    var func = this._stack.pop();

    var res = func + "(" + argList + ")";

    this._stack.push(res);

    return node;
  }

  VisitCall(node: CallExpression): Expression {
    var res = "";

    if (node.obj !== null) {
      this.Visit(node.obj);
      res = this._stack.pop() + ".";
    }

    this.VisitMany(node.args);

    var n = node.args.length;
    var args = new Array<string>(n);
    for (var i = 0; i < n; i++) {
      args[n - i - 1] = this._stack.pop();
    }

    var argList = args.join(", ");

    res += node.method + "(" + argList + ")";

    this._stack.push(res);

    return node;
  }

  VisitNew(node: NewExpression): Expression {
    this.VisitMany(node.args);

    var n = node.args.length;
    var args = new Array<string>(n);
    for (var i = 0; i < n; i++) {
      args[n - i - 1] = this._stack.pop();
    }

    var argList = args.join(", ");

    var res = "new " + node.type + "(" + argList + ")";

    this._stack.push(res);

    return node;
  }

  VisitMember(node: MemberExpression): Expression {
    var res = "";

    if (node.obj !== null) {
      this.Visit(node.obj);
      res = this._stack.pop() + ".";
    }

    res += node.member;

    this._stack.push(res);

    return node;
  }

  VisitIndex(node: IndexExpression): Expression {
    this.Visit(node.obj);
    var res = this._stack.pop();

    this.VisitMany(node.args);

    var n = node.args.length;
    var args = new Array<string>(n);
    for (var i = 0; i < n; i++) {
      args[n - i - 1] = this._stack.pop();
    }

    var argList = args.join(", ");

    res += "[" + argList + "]";

    this._stack.push(res);

    return node;
  }
}

class FreeVariableScanner extends ExpressionVisitor {
  _stack: Expression[][];
  _result: Expression[];

  constructor() {
    super();
    this._stack = new Array<Expression[]>();
    this._result = new Array<Expression>();
  }

  get result(): Expression[] {
    return this._result;
  }

  VisitParameter(node: ParameterExpression): Expression {
    var found = false;

    for (var i = this._stack.length - 1; i >= 0; i--) {
      if (this._stack[i].indexOf(node) >= 0) {
        found = true;
        break;
      }
    }

    if (!found) {
      this._result.push(node);
    }

    return node;
  }

  VisitLambda<T extends Function>(node: LambdaExpression<T>): Expression {
    this._stack.push(node.parameters);

    this.Visit(node.body);

    this._stack.pop();

    return node;
  }
}

enum ExpressionType {
  Constant,
  Parameter,
  Lambda,
  Add,
  Subtract,
  Multiply,
  Divide,
  Modulo,
  And,
  Or,
  AndAlso,
  OrElse,
  ExclusiveOr,
  Equal,
  NotEqual,
  LessThan,
  LessThanOrEqual,
  GreaterThan,
  GreaterThanOrEqual,
  LeftShift,
  RightShift,
  Invoke,
  Not,
  Negate,
  UnaryPlus,
  OnesComplement,
  Condition,
  New,
  Call,
  Member,
  Index,
}

class Binder extends ExpressionVisitor {
  _stack: Expression[][];
  _resources: any;

  constructor(resources: any) {
    super();
    this._stack = new Array<Expression[]>();
    this._resources = resources;
  }

  VisitParameter(node: ParameterExpression): Expression {
    var found = false;

    for (var i = this._stack.length - 1; i >= 0; i--) {
      if (this._stack[i].indexOf(node) >= 0) {
        found = true;
        break;
      }
    }

    if (!found) {
      return Expression.Constant(this._resources[node.name]);
    }

    return node;
  }

  VisitLambda<T extends Function>(node: LambdaExpression<T>): Expression {
    this._stack.push(node.parameters);

    this.Visit(node.body);

    this._stack.pop();

    return node;
  }
}

var resources =
  {
    "my://xs": [1, 2, 3, 4, 5],
    "my://ss": ["bar", "foo", "qux"],
    "rx://operators/filter": function (xs: any[], f: (any) => boolean) { return xs.filter(f); },
    "rx://operators/map": function (xs: any[], f: (any) => any) { return xs.map(f); },
  };

var x = Expression.Parameter("x");
var f1 =
  Expression.Invoke(
    Expression.Parameter("rx://operators/map"),
    Expression.Invoke(
      Expression.Parameter("rx://operators/filter"),
      Expression.Parameter("my://xs"),
      Expression.Lambda<(number) => boolean>(
        Expression.Equal(
          Expression.Modulo(
            x,
            Expression.Constant(2)
            ),
          Expression.Constant(0)
          ),
        x
        )
      ),
    Expression.Lambda<(number) => boolean>(
      Expression.Multiply(
        x,
        x
        ),
      x
      )
    );

var f2 =
  Expression.Invoke(
    Expression.Parameter("rx://operators/map"),
    Expression.Parameter("my://ss"),
    Expression.Lambda<(string) => string>(
      Expression.Call(
        x,
        "substring",
        Expression.Constant(1)
        ),
      x
      )
    );

var binder = new Binder(resources);

var b1 = Expression.Lambda<() => number[]>(binder.Visit(f1));
var c1 = b1.Compile();
var r1 = c1();
alert(r1.join(", "));

var b2 = Expression.Lambda<() => string[]>(binder.Visit(f2));
var c2 = b2.Compile();
var r2 = c2();
alert(r2.join(", "));
