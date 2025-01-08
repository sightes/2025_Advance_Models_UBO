.PHONY: all clean

# Variables
PROJECT = $(project)
LANGUAGE = $(language)

# Reglas principales
all: $(LANGUAGE)

clean:
	rm -rf $(PROJECT)
	@echo "Directorio $(PROJECT) eliminado."

# Función para determinar la estructura según el lenguaje
define create_structure
    @echo "Creando estructura para $(LANGUAGE)..."
    mkdir -p $(1)
    $(2)
    @echo "Estructura de directorios y archivos para $(LANGUAGE) creada."
endef

Python:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/main.py \
	    && mkdir -p $(PROJECT)/tests \
	    && echo "# $(PROJECT)" > $(PROJECT)/README.md \
	)

R:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/main.R \
	    && mkdir -p $(PROJECT)/data $(PROJECT)/output $(PROJECT)/docs \
	    && echo "# $(PROJECT)" > $(PROJECT)/README.md \
	)

Scala:
	$(call create_structure,$(PROJECT)/src/main/scala,\
	    touch $(PROJECT)/src/main/scala/Main.scala \
	    && mkdir -p $(PROJECT)/src/test/scala $(PROJECT)/src/main/resources \
	    && echo "name := \"$(PROJECT)\"" > $(PROJECT)/build.sbt \
	    && echo "version := \"0.1.0\"" >> $(PROJECT)/build.sbt \
	    && echo "scalaVersion := \"2.13.12\"" >> $(PROJECT)/build.sbt \
	    && echo "object Main {" > $(PROJECT)/src/main/scala/Main.scala \
	    && echo "  def main(args: Array[String]): Unit = {" >> $(PROJECT)/src/main/scala/Main.scala \
	    && echo "    println(\"Hello, $(PROJECT)!\")" >> $(PROJECT)/src/main/scala/Main.scala \
	    && echo "  }" >> $(PROJECT)/src/main/scala/Main.scala \
	    && echo "}" >> $(PROJECT)/src/main/scala/Main.scala \
	)

Rust:
	$(call create_structure,$(PROJECT)/src,\
	    touch $(PROJECT)/src/main.rs \
	    && echo "[package]" > $(PROJECT)/Cargo.toml \
	    && echo "name = \"$(PROJECT)\"" >> $(PROJECT)/Cargo.toml \
	    && echo "version = \"0.1.0\"" >> $(PROJECT)/Cargo.toml \
	    && echo "edition = \"2021\"" >> $(PROJECT)/Cargo.toml \
	    && echo "fn main() {" > $(PROJECT)/src/main.rs \
	    && echo "    println!(\"Hello, $(PROJECT)!\");" >> $(PROJECT)/src/main.rs \
	    && echo "}" >> $(PROJECT)/src/main.rs \
	)

C++:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/main.cpp \
	    && mkdir -p $(PROJECT)/include $(PROJECT)/src \
	    && echo "# $(PROJECT)" > $(PROJECT)/README.md \
	    && echo "#include <iostream>" > $(PROJECT)/main.cpp \
	    && echo "int main() {" >> $(PROJECT)/main.cpp \
	    && echo "    std::cout << \"Hello, $(PROJECT)!\" << std::endl;" >> $(PROJECT)/main.cpp \
	    && echo "    return 0;" >> $(PROJECT)/main.cpp \
	    && echo "}" >> $(PROJECT)/main.cpp \
	)

Java:
	$(call create_structure,$(PROJECT)/src/main/java,\
	    touch $(PROJECT)/src/main/java/Main.java \
	    && mkdir -p $(PROJECT)/src/test/java \
	    && echo "public class Main {" > $(PROJECT)/src/main/java/Main.java \
	    && echo "    public static void main(String[] args) {" >> $(PROJECT)/src/main/java/Main.java \
	    && echo "        System.out.println(\"Hello, $(PROJECT)!\");" >> $(PROJECT)/src/main/java/Main.java \
	    && echo "    }" >> $(PROJECT)/src/main/java/Main.java \
	    && echo "}" >> $(PROJECT)/src/main/java/Main.java \
	)

Javascript:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/index.js \
	    && mkdir -p $(PROJECT)/tests \
	    && echo "# $(PROJECT)" > $(PROJECT)/README.md \
	    && echo "console.log('Hello, $(PROJECT)!');" > $(PROJECT)/index.js \
	)

SQL:
	$(call create_structure,$(PROJECT)/sql,\
	    touch $(PROJECT)/sql/query.sql \
	    && echo "-- SQL queries for $(PROJECT)" > $(PROJECT)/sql/query.sql \
	)

SPARK:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/main.py \
	    && mkdir -p $(PROJECT)/data $(PROJECT)/output $(PROJECT)/notebooks \
	    && echo "# Spark Project $(PROJECT)" > $(PROJECT)/README.md \
	    && echo "from pyspark.sql import SparkSession" > $(PROJECT)/main.py \
	    && echo "spark = SparkSession.builder.appName('$(PROJECT)').getOrCreate()" >> $(PROJECT)/main.py \
	    && echo "print('Spark session created for $(PROJECT)')" >> $(PROJECT)/main.py \
	)

Perl:
	$(call create_structure,$(PROJECT),\
	    touch $(PROJECT)/main.pl \
	    && echo "#!/usr/bin/perl" > $(PROJECT)/main.pl \
	    && echo "use strict;" >> $(PROJECT)/main.pl \
	    && echo "use warnings;" >> $(PROJECT)/main.pl \
	    && echo "print \"Hello, $(PROJECT)!\\n\";" >> $(PROJECT)/main.pl \
	    && chmod +x $(PROJECT)/main.pl \
	)