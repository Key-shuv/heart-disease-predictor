const CACHE_NAME = "heart-predictor-v2";

const urlsToCache = [
  "/",
  "/static/style.css",
  "/static/manifest.json",
  "/static/icons/web-app-manifest-192x192.png",
  "/static/icons/web-app-manifest-512x512.png"
];


// Install service worker
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
  );

  self.skipWaiting();
});


// Activate new service worker
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(cacheName => cacheName !== CACHE_NAME)
          .map(cacheName => caches.delete(cacheName))
      );
    })
  );

  self.clients.claim();
});


// Serve cached files when available
self.addEventListener("fetch", event => {

  // Only handle GET requests
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {

        if (cachedResponse) {
          return cachedResponse;
        }

        return fetch(event.request);
      })
  );

});
