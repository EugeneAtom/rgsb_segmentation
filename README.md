# rgsb_segmentation

## Использование
0) скачать модель https://disk.yandex.com/d/3ww66g0zkP4S5A и поместить её в папку model, созданную рядом с app
1) скачать образ - docker pull efrolov/rgsb_vectorize_segment:3
2) запуск - docker run -p 5090:5090 efrolov/rgsb_vectorize_segment:3
3) На http://localhost:500/ будет запущен поисковый сервис. Автодокументация доступна по http://localhost:5000/docs
4) На странице автодокументации выбрать vectorize, затем Try it out, затем выбрать картинку и запустить.
