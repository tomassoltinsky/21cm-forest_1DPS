
#######################################################################
#  Look at end of file for a brief guide to any compile-time options. #
#######################################################################

#--------------------------------------- Options
#OPTS +=-DNOPECVEL
#OPTS +=-DTAUWEIGHT

#--------------------------------------- Select system
SYSTYPE="tsoltins"
#SYSTYPE="brahan"


ifeq ($(SYSTYPE),"tsoltins")
CC = gcc -fopenmp	
OPTIMIZE = -O3
CFLAGS =  $(OPTIMIZE) -Wall -Wno-unused-result
CFLAGS += $(OPTS)	
endif

ifeq ($(SYSTYPE),"brahan")
CC = gcc	
OPTIMIZE = -O3
CFLAGS =  $(OPTIMIZE) -Wall
CFLAGS += $(OPTS)	
endif

EXEC = LosExtract
OBJS = extract_los.o 
INCL = proto.h global_vars.h
LIBS = -lm


$(EXEC): $(OBJS)
	$(CC) $(OBJS) $(LIBS) -o $(EXEC)
$(OBJS): $(INCL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXEC)

##############################################################################
#
# Line-of-sight extraction options:
#
#	- NOPECVEL 	Computes the spectra ignoring gas peculiar velocities
#
##############################################################################