import { Schema } from '../lib';

test('boolean type', () => {
  expect(Schema.BOOLEAN.inputString).toEqual('boolean');
  expect(Schema.BOOLEAN.isPrimitive).toEqual(true);
});

test('binary type', () => {
  expect(Schema.BINARY.inputString).toEqual('binary');
  expect(Schema.BINARY.isPrimitive).toEqual(true);
});

test('bigint type', () => {
  expect(Schema.BIG_INT.inputString).toEqual('bigint');
  expect(Schema.BIG_INT.isPrimitive).toEqual(true);
});

test('double type', () => {
  expect(Schema.DOUBLE.inputString).toEqual('double');
  expect(Schema.DOUBLE.isPrimitive).toEqual(true);
});

test('float type', () => {
  expect(Schema.FLOAT.inputString).toEqual('float');
  expect(Schema.FLOAT.isPrimitive).toEqual(true);
});

test('integer type', () => {
  expect(Schema.INTEGER.inputString).toEqual('int');
  expect(Schema.INTEGER.isPrimitive).toEqual(true);
});

test('smallint type', () => {
  expect(Schema.SMALL_INT.inputString).toEqual('smallint');
  expect(Schema.SMALL_INT.isPrimitive).toEqual(true);
});

test('tinyint type', () => {
  expect(Schema.TINY_INT.inputString).toEqual('tinyint');
  expect(Schema.TINY_INT.isPrimitive).toEqual(true);
});

test('decimal type', () => {
  expect(Schema.decimal(16).inputString).toEqual('decimal(16)');
  expect(Schema.decimal(16, 1).inputString).toEqual('decimal(16,1)');
  expect(Schema.decimal(16).isPrimitive).toEqual(true);
  expect(Schema.decimal(16, 1).isPrimitive).toEqual(true);
});
// TODO: decimal bounds

test('date type', () => {
  expect(Schema.DATE.inputString).toEqual('date');
  expect(Schema.DATE.isPrimitive).toEqual(true);
});

test('timestamp type', () => {
  expect(Schema.TIMESTAMP.inputString).toEqual('timestamp');
  expect(Schema.TIMESTAMP.isPrimitive).toEqual(true);
});

test('string type', () => {
  expect(Schema.STRING.inputString).toEqual('string');
  expect(Schema.STRING.isPrimitive).toEqual(true);
});

test('char type', () => {
  expect(Schema.char(1).inputString).toEqual('char(1)');
  expect(Schema.char(1).isPrimitive).toEqual(true);
});

test('char length must be test(at least 1', () => {
  expect(() => Schema.char(1)).not.toThrow();
  expect(() => Schema.char(0)).toThrow();
  expect(() => Schema.char(-1)).toThrow();
});

test('char length must test(be <= 255', () => {
  expect(() => Schema.char(255)).not.toThrow();
  expect(() => Schema.char(256)).toThrow();
});

test('varchar type', () => {
  expect(Schema.varchar(1).inputString).toEqual('varchar(1)');
  expect(Schema.varchar(1).isPrimitive).toEqual(true);
});

test('varchar length must be test(at least 1', () => {
  expect(() => Schema.varchar(1)).not.toThrow();
  expect(() => Schema.varchar(0)).toThrow();
  expect(() => Schema.varchar(-1)).toThrow();
});

test('varchar length must test(be <= 65535', () => {
  expect(() => Schema.varchar(65535)).not.toThrow();
  expect(() => Schema.varchar(65536)).toThrow();
});

test('test(array<string>', () => {
  const type = Schema.array(Schema.STRING);
  expect(type.inputString).toEqual('array<string>');
  expect(type.isPrimitive).toEqual(false);
});

test('array<test(char(1)>', () => {
  const type = Schema.array(Schema.char(1));
  expect(type.inputString).toEqual('array<char(1)>');
  expect(type.isPrimitive).toEqual(false);
});

test('test(array<array>', () => {
  const type = Schema.array(
    Schema.array(Schema.STRING));
  expect(type.inputString).toEqual('array<array<string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('test(array<map>', () => {
  const type = Schema.array(
    Schema.map(Schema.STRING, Schema.STRING));
  expect(type.inputString).toEqual('array<map<string,string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('test(array<struct>', () => {
  const type = Schema.array(
    Schema.struct([{
      name: 'key',
      type: Schema.STRING,
    }]));
  expect(type.inputString).toEqual('array<struct<key:string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<test(string,string>', () => {
  const type = Schema.map(
    Schema.STRING,
    Schema.STRING,
  );
  expect(type.inputString).toEqual('map<string,string>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<test(int,string>', () => {
  const type = Schema.map(
    Schema.INTEGER,
    Schema.STRING,
  );
  expect(type.inputString).toEqual('map<int,string>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<char(1),test(char(1)>', () => {
  const type = Schema.map(
    Schema.char(1),
    Schema.char(1),
  );
  expect(type.inputString).toEqual('map<char(1),char(1)>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<test(string,array>', () => {
  const type = Schema.map(
    Schema.char(1),
    Schema.array(Schema.STRING),
  );
  expect(type.inputString).toEqual('map<char(1),array<string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<test(string,map>', () => {
  const type = Schema.map(
    Schema.char(1),
    Schema.map(
      Schema.STRING,
      Schema.STRING),
  );
  expect(type.inputString).toEqual('map<char(1),map<string,string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('map<test(string,struct>', () => {
  const type = Schema.map(
    Schema.char(1),
    Schema.struct([{
      name: 'key',
      type: Schema.STRING,
    }]),
  );
  expect(type.inputString).toEqual('map<char(1),struct<key:string>>');
  expect(type.isPrimitive).toEqual(false);
});

test('map throws if keyType is test(non-primitive', () => {
  expect(() => Schema.map(
    Schema.array(Schema.STRING),
    Schema.STRING,
  )).toThrow();
  expect(() => Schema.map(
    Schema.map(Schema.STRING, Schema.STRING),
    Schema.STRING,
  )).toThrow();
  expect(() => Schema.map(
    Schema.struct([{
      name: 'key',
      type: Schema.STRING,
    }]),
    Schema.STRING,
  )).toThrow();
});

test('struct type', () => {
  const type = Schema.struct([{
    name: 'primitive',
    type: Schema.STRING,
  }, {
    name: 'with_comment',
    type: Schema.STRING,
    comment: 'this has a comment',
  }, {
    name: 'array',
    type: Schema.array(Schema.STRING),
  }, {
    name: 'map',
    type: Schema.map(Schema.STRING, Schema.STRING),
  }, {
    name: 'nested_struct',
    type: Schema.struct([{
      name: 'nested',
      type: Schema.STRING,
      comment: 'nested comment',
    }]),
  }]);

  expect(type.isPrimitive).toEqual(false);
  expect(type.inputString).toEqual(

    'struct<primitive:string,with_comment:string,array:array<string>,map:map<string,string>,nested_struct:struct<nested:string>>',
  );
});
