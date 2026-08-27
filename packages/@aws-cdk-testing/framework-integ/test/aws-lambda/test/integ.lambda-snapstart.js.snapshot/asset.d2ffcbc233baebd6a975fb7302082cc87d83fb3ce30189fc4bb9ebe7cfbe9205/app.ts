
exports.handler = async (event: any) => {
  console.log('hello world');
  console.log(`event ${JSON.stringify(event)}`);
  return {
    statusCode: 200,
  };
};
