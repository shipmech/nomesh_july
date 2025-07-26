import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from a_config import SimulationConfig
from g_datamodule import FluidDynamicsDataModule
from f_simulation import FluidSimulation

def main():
    config = SimulationConfig()
    data_module = FluidDynamicsDataModule(config)
    simulation = FluidSimulation(config)
    
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator=config.device,
        devices='auto',
        enable_progress_bar=True,
        log_every_n_steps=2,
        logger=TensorBoardLogger(save_dir="logs", name="fluid_simulation"),
        gradient_clip_val=1.0,  # Add gradient clipping
    )
    
    trainer.fit(simulation, data_module)

if __name__ == "__main__":
    main()

