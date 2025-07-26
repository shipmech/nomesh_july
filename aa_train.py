import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from a_config import SimulationConfig
from g_datamodule import FluidDynamicsDataModule  # Updated import
from f_simulation import FluidSimulation  # Use current simulation

def main():
    config = SimulationConfig()
    data_module = FluidDynamicsDataModule(config)
    simulation = FluidSimulation(config)
    
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator=config.device,  # Updated from config.device for flexibility
        devices='auto',
        enable_progress_bar=True,
        log_every_n_steps=2,
        logger=TensorBoardLogger(save_dir="logs", name="fluid_simulation"),
        gradient_clip_val=1.0,
        profiler=SimpleProfiler(filename="perf_logs"),
        #detect_anomaly=True,
    )
    
    trainer.fit(simulation, data_module)

if __name__ == "__main__":
    main()