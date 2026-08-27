"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handler = async (event) => {
    console.log('hello world');
    console.log(`event ${JSON.stringify(event)}`);
    return {
        statusCode: 200,
    };
};
