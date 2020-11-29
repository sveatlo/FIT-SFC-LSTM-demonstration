.PHONY=run all build pack docs banner

# build variables
CFLAGS=-std=c++11  -lstdc++ -Wall -Wextra -O2 -lm
CC=gcc
## which modules should be build
MODULES=log matrix lstm
OBJECT_FILE_PATTERN=$(DIST_DIR)%.o
SRC_DIR=src/
DIST_DIR=dist/
DOCS_DIR=docs/
BINARY_NAME=lstm-nn
ARCHIVEFILENAME=xhanze10.tar

# documentation variables
DOCS_SOURCES=$(DOCS_DIR)manual/documentation.tex $(DOCS_DIR)manual/czechiso.bst \
	$(DOCS_DIR)manual/references.bib $(DOCS_DIR)manual/Makefile $(DOCS_DIR)manual/images
PDF_FILENAME=manual.pdf

all: build $(DIST_DIR)$(BINARY_NAME) banner

banner:
	@echo "██╗     ███████╗████████╗███╗   ███╗    ███╗   ██╗███╗   ██╗"
	@echo "██║     ██╔════╝╚══██╔══╝████╗ ████║    ████╗  ██║████╗  ██║"
	@echo "██║     ███████╗   ██║   ██╔████╔██║    ██╔██╗ ██║██╔██╗ ██║"
	@echo "██║     ╚════██║   ██║   ██║╚██╔╝██║    ██║╚██╗██║██║╚██╗██║"
	@echo "███████╗███████║   ██║   ██║ ╚═╝ ██║    ██║ ╚████║██║ ╚████║"
	@echo "╚══════╝╚══════╝   ╚═╝   ╚═╝     ╚═╝    ╚═╝  ╚═══╝╚═╝  ╚═══╝"
	@echo " _                                 _   _       "
	@echo "| |__  _   _   _____   _____  __ _| |_| | ___  "
	@echo "| '_ \\| | | | / __\\ \\ / / _ \\/ _\` | __| |/ _ \\ "
	@echo "| |_) | |_| | \\__ \\\\ V /  __/ (_| | |_| | (_) |"
	@echo "|_.__/ \\__, | |___/ \\_/ \\___|\\__,_|\\__|_|\\___/ "
	@echo "       |___/                                   "


dev: build

documentation: $(wildcard $(SRC_DIR)*) $(DOCS_SOURCES)
	doxygen
	OUTPUT_PDF=$(PDF_FILENAME) make -C $(DOCS_DIR)manual

stats:
	@echo -n "Lines of code: " && wc -l $(wildcard $(SRC_DIR)*.cpp $(SRC_DIR)*.h) | tail -n 1 | sed -r "s/[ ]*([0-9]+).*/\1/g"
	@echo -n "Size of code: " && du -hsc $(wildcard $(SRC_DIR)*.cpp $(SRC_DIR)*.h) | tail -n 1 | cut -f 1


# Link all the modules together
build: $(DIST_DIR)$(BINARY_NAME)

build-prod: build
	mv $(DIST_DIR)$(BINARY_NAME) ./$(BINARY_NAME)

# Build binary
$(DIST_DIR)$(BINARY_NAME): $(SRC_DIR)main.cpp $(patsubst %,$(OBJECT_FILE_PATTERN), $(MODULES))
	$(CC) $(CFLAGS) \
		$(SRC_DIR)main.cpp $(patsubst %,$(OBJECT_FILE_PATTERN), $(MODULES)) \
	-o $(DIST_DIR)$(BINARY_NAME)

# Make modules independently
$(OBJECT_FILE_PATTERN): $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)$*.cpp -o $(DIST_DIR)$*.o

run: build
	exec $(DIST_DIR)$(BINARY_NAME)
pack: $(SRC_DIR)*.cpp $(SRC_DIR)*.h $(DOCS_SOURCES) Makefile Doxyfile
	make documentation
	mv docs/manual/$(PDF_FILENAME) $(PDF_FILENAME)
	make clean
	tar cf $(ARCHIVEFILENAME) $(SRC_DIR) $(DIST_DIR) $(DOCS_DIR) $(PDF_FILENAME) Makefile Doxyfile README.md
clean:
	make -C $(DOCS_DIR)manual clean
	rm -rf ./*.o $(DIST_DIR)$(BINARY_NAME) $(DIST_DIR)*.o $(DIST_DIR)*.out $(DIST_DIR)*.a $(DIST_DIR)*.so $(SRC_DIR)*.gch \
			$(ARCHIVEFILENAME) $(DOCS_DIR)doxygen \
			$(filter-out $(DOCS_SOURCES) , $(wildcard $(DOCS_DIR)manual/*))
