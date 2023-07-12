
.PHONY: run dockerbuild dockerrun

run:
	@uvicorn captioner:app --reload

dockerbuild:
	@docker build -t captioner .

dockerrun:
	@docker run --rm -p 8000:8000 captioner

test: 
	@time curl -X POST -H "Content-Type: application/json" -d '{"url": "https://cataas.com/cat" }'  http://127.0.0.1:8000/caption/
