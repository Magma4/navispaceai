# syntax=docker/dockerfile:1
# Multi-stage frontend image: build with Node, serve static files with Nginx.

FROM node:20-alpine AS builder
WORKDIR /app

# Install dependencies first for cache efficiency.
COPY frontend/package*.json /app/
RUN npm ci

# Copy frontend source and build.
COPY frontend /app

# Build-time API URL injection for Vite/React env usage.
ARG VITE_API_URL=http://localhost:8000
ARG REACT_APP_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

RUN npm run build

FROM nginx:1.27-alpine AS runtime

# Configure Nginx to serve SPA on port 3000.
RUN rm -f /etc/nginx/conf.d/default.conf && \
    printf 'server {\n  listen 3000;\n  server_name _;\n  root /usr/share/nginx/html;\n  index index.html;\n  location / {\n    try_files $uri /index.html;\n  }\n}\n' > /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
