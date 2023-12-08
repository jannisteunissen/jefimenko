# Use gfortran unless already defined
F90 := gfortran
FFLAGS := -O2 -g -Wall -Wextra -fcheck=all

OBJS := m_config.o
EXAMPLES := jefimenko

.PHONY:	all test clean

all: 	$(LIB) $(EXAMPLES)

$(LIB): $(OBJS)
	$(RM) $@
	$(AR) rcs $@ $^

clean:
	$(RM) $(EXAMPLES) m_config.o m_config.mod $(LIB)

# Dependency information
$(EXAMPLES): m_config.o

# How to get .o object files from .f90 source files
%.o: %.f90
	$(F90) -c -o $@ $< $(FFLAGS)

# How to get executables from .o object files
%: %.o
	$(F90) -o $@ $^ $(FFLAGS)
