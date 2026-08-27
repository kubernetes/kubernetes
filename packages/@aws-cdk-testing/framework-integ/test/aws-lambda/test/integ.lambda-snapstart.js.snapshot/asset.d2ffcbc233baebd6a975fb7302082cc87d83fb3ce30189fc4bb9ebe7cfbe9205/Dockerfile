FROM public.ecr.aws/lambda/nodejs:18
ARG FUNCTION_DIR="/var/task"
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}
# Copy handler function and package.json
COPY app.js ${FUNCTION_DIR}
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
