install-server:
	cd server && make install

install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then cd server/custom_kernels && python setup.py install; else echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; fi

install-integration-tests:
	cd integration-tests && pip install -r requirements.txt
	cd clients/python && pip install .

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

install-benchmark:
	cd benchmark && cargo install --path .

install: install-server install-router install-launcher install-custom-kernels

server-dev:
	cd server && make run-dev --host 0.0.0.0 

router-dev:
	cd router && cargo run -- --host 0.0.0.0 

rust-tests: install-router install-launcher
	cargo test

integration-tests: install-integration-tests
	pytest -s -vv -m "not private" integration-tests

update-integration-tests: install-integration-tests
	pytest -s -vv --snapshot-update integration-tests

python-server-tests:
	HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv -m "not private" server/tests

python-client-tests:
	pytest clients/python/tests

python-tests: python-server-tests python-client-tests

run-tiny-llama:
	text-generation-launcher --model-id  benchang1110/Taiwan-tinyllama-v1.0-chat --hostname 0.0.0.0 --port 8080

run-falcon-7b-instruct-quantize:
	text-generation-launcher --model-id tiiuae/falcon-7b-instruct --quantize bitsandbytes --port 8080

clean:
	rm -rf target aml